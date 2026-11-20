"""单进程解码 API（FastAPI）。

端点
----
- POST /branches              提交 token ID 建立上下文
- POST /branches/{id}/fork    从任一已有 token 处分叉（零拷贝继承 KV）
- POST /branches/{id}/tokens  批量追加 token（超限整批拒绝）
- DELETE /branches/{id}       关闭分支（最后一个引用关闭后释放 KV）
- GET  /branches              列出分支与内部引用计数（调试）
- GET  /stats                 缓存用量
- GET  /docs                  Swagger UI

所有成功响应都带缓存用量；建立上下文 / 分叉 / 追加还返回与整段重算
（model.forward_full）一致的下一 token logits。
"""

from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from .model import (
    DEFAULT_D_MODEL,
    DEFAULT_SEED,
    DEFAULT_VOCAB_SIZE,
    build_model,
)
from .store import AppendComputeError, BranchStore, CapacityExceeded


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


VOCAB_SIZE = _env_int("MODEL_VOCAB_SIZE", DEFAULT_VOCAB_SIZE)
D_MODEL = _env_int("MODEL_D_MODEL", DEFAULT_D_MODEL)
SEED = _env_int("MODEL_SEED", DEFAULT_SEED)
MAX_CACHED_TOKENS = _env_int("MAX_CACHED_TOKENS", 128)

MODEL = build_model(seed=SEED, vocab_size=VOCAB_SIZE, d_model=D_MODEL)
STORE = BranchStore(MODEL, max_cached_tokens=MAX_CACHED_TOKENS)


# --------------------------------------------------------------------- 请求模型


class CreateBranchRequest(BaseModel):
    token_ids: list[int] = Field(
        ..., description="用于建立上下文的 token ID 序列；可为空数组"
    )

    @field_validator("token_ids")
    @classmethod
    def _check_range(cls, value: list[int]) -> list[int]:
        for tid in value:
            if tid < 0 or tid >= VOCAB_SIZE:
                raise ValueError(f"token id {tid} out of range [0, {VOCAB_SIZE - 1}]")
        return value


class ForkRequest(BaseModel):
    offset: int | None = Field(
        default=None,
        description="保留父分支前 offset 个 token；省略或 null 表示继承完整上下文",
    )

    @field_validator("offset")
    @classmethod
    def _non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("offset must be >= 0")
        return value


class AppendTokensRequest(BaseModel):
    token_ids: list[int] = Field(..., min_length=1, description="本批追加的 token ID")

    @field_validator("token_ids")
    @classmethod
    def _check_range(cls, value: list[int]) -> list[int]:
        for tid in value:
            if tid < 0 or tid >= VOCAB_SIZE:
                raise ValueError(f"token id {tid} out of range [0, {VOCAB_SIZE - 1}]")
        return value


# --------------------------------------------------------------------- 响应模型


class BranchResponse(BaseModel):
    branch_id: str
    token_ids: list[int]
    length: int
    logits: list[float] | None
    full_recompute_logits: list[float] | None
    usage: dict


def _make_response(
    branch,
    logits,
    full_recompute_logits=None,
) -> BranchResponse:
    """构造响应。

    ``logits`` / ``full_recompute_logits`` 由 store 在写入前一并算好并校验
    一致；这里只做 ndarray -> list 转换，绝不在状态写入之后再做可能失败的
    重算（否则失败重试会把同一批 token 拼两遍）。
    """
    return BranchResponse(
        branch_id=branch.id,
        token_ids=list(branch.token_ids),
        length=len(branch.token_ids),
        logits=None if logits is None else np.asarray(logits).tolist(),
        full_recompute_logits=(
            None
            if full_recompute_logits is None
            else np.asarray(full_recompute_logits).tolist()
        ),
        usage=STORE.usage(),
    )


# --------------------------------------------------------------------- 应用


def create_app() -> FastAPI:
    app = FastAPI(
        title="NumPy 单头因果注意力解码 API",
        version="1.0.0",
        description=(
            "单进程解码服务：提交 token 建立上下文、任意位置分叉（继承 KV）、"
            "批量追加与关闭。缓存上限按实际保存的 token 位置数计。"
        ),
    )

    @app.exception_handler(CapacityExceeded)
    async def _capacity_handler(_request: Request, exc: CapacityExceeded):
        # 413：整批已被拒绝，分支与占用量均未改变。
        return JSONResponse(
            status_code=413,
            content={
                "error": "capacity_exceeded",
                "detail": str(exc),
                "usage": STORE.usage(),
            },
        )

    @app.exception_handler(AppendComputeError)
    async def _append_compute_handler(_request: Request, exc: AppendComputeError):
        # 500：增量推理 / 整段重算 / 一致性校验失败。store 保证失败发生在
        # 任何状态写入之前，分支 token、KV 引用与缓存用量均保持原值；
        # 客户端可安全重试同一批次，不会重复拼接。
        return JSONResponse(
            status_code=500,
            content={
                "error": "append_compute_failed",
                "detail": str(exc),
                "usage": STORE.usage(),
            },
        )

    # 注意：以下处理阻塞型同步 store 的端点全部用普通 def（而非 async def）。
    # Starlette 会把同步端点放到 anyio 工作线程池中执行，使长批次的 CPU 计算
    # 不会卡住唯一的事件循环线程——其他分支的追加与关闭因此能并发进行。
    # 线程安全与原子性由 BranchStore 内部的全局锁与分支锁保证。
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health():
        return {"status": "ok", "vocab_size": VOCAB_SIZE, "d_model": D_MODEL}

    @app.get("/stats")
    def stats():
        return STORE.usage()

    @app.get("/branches")
    def list_branches():
        return STORE.debug_snapshot()

    @app.post("/branches", status_code=201, response_model=BranchResponse)
    def create_branch(req: CreateBranchRequest):
        branch, logits, full_logits = STORE.create_branch(req.token_ids)
        return _make_response(branch, logits, full_logits)

    @app.post("/branches/{branch_id}/fork", response_model=BranchResponse)
    def fork_branch(branch_id: str, req: ForkRequest):
        try:
            branch, logits, full_logits = STORE.fork(branch_id, req.offset)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": "branch_not_found", "branch_id": branch_id},
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": "invalid_offset", "detail": str(exc)},
            )
        return _make_response(branch, logits, full_logits)

    @app.post("/branches/{branch_id}/tokens", response_model=BranchResponse)
    def append_tokens(branch_id: str, req: AppendTokensRequest):
        try:
            branch, logits, full_logits = STORE.append(branch_id, req.token_ids)
        except KeyError:
            # 分支不存在，或在途追加的提交阶段发现分支已被关闭。
            return JSONResponse(
                status_code=404,
                content={"error": "branch_not_found", "branch_id": branch_id},
            )
        return _make_response(branch, logits, full_logits)

    @app.delete("/branches/{branch_id}")
    def close_branch(branch_id: str):
        try:
            STORE.close(branch_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": "branch_not_found", "branch_id": branch_id},
            )
        return {"branch_id": branch_id, "closed": True, "usage": STORE.usage()}

    return app


app = create_app()
