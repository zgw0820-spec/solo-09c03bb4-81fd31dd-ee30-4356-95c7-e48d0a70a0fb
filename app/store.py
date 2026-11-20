"""分支与 KV 缓存存储。

设计要点
--------
- 每个分支（:class:`Branch`）持有自己的 token 序列，并指向一个 :class:`KvState`。
- :class:`KvState` 是一份物理 K/V 数据（NumPy 数组），通过引用计数在分支间
  共享；分叉是零拷贝的——新分支直接指向同一份 KV，只做引用计数 +1。
- 追加 token 时若当前 KV 被多个分支引用，则先做写时复制（copy-on-write），
  保证“各分支只使用自己的上下文”，追加不会影响共享同一份 KV 的其他分支。
- 分支可以从任一已有 token 处分叉（offset <= 当前长度）：截断 fork 复用底层
  物理数组，只按自己的视图长度读取，不产生新的缓存占用。
- 容量按“实际保存的 token 位置数”计量，即所有仍存活的物理 KvState 的
  有效长度之和；共享不重复计数，截断不占额外空间。
- 容量超限的提交在改动任何状态之前完成检查，整批拒绝（原子失败），
  原分支内容与全局占用量保持不变。
- 每份 KV 在最后一个引用它的分支关闭后释放（引用计数归零即删除）。

追加的两阶段原子语义（本模块核心）
----------------------------------
一次批量追加分为“计算”和“提交”两段：

1. **计算阶段（锁外，可并发）**：在本地副本上依次完成
   增量推理（:func:`model.step`）与整段重算（:func:`model.forward_full`），
   并核对两份 logits 一致。此阶段不触碰任何活状态——读 KV 只做切片拷贝，
   ``model.step`` 本身也是纯函数（每次返回新数组）。任一计算/校验失败，
   分支的 token、KV 引用与全局缓存用量都不会改变。
2. **提交阶段（全局锁内，原子）**：重新读取分支与 KV 的最新共用情况
   （计算期间可能有别的线程 fork/close），据此重新核算写后的占用增量；
   超限时整批不写入并抛 :class:`CapacityExceeded`；分支若已被关闭则
   抛 :class:`KeyError`（API 层映射为 404）。只有全部检查通过才一次性
   切换 KV 引用、追加 token 并更新占用量。

线程模型
--------
- 一把全局可重入锁（:attr:`BranchStore._lock`）只保护短暂的状态检查与
  提交，重计算在锁外进行，因此一个分支的长批次计算不会阻塞其他分支的
  续写与关闭。
- 每个分支另有一把独立锁（:attr:`BranchStore._branch_locks`），串行化
  针对同一分支的并发追加，使成功批次严格按获取顺序逐批接上。
- 分支锁不与全局锁嵌套持有：先取分支锁做计算，提交时只短暂获取全局锁，
  不存在多锁获取顺序导致的死锁。
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

from . import model as model_mod

# logits 一致性校验容差（与测试、e2e 脚本保持一致）。
LOGITS_RTOL = 1e-5
LOGITS_ATOL = 1e-6


class CapacityExceeded(Exception):
    """追加整批 token 后会超出缓存上限。"""


class AppendComputeError(Exception):
    """追加的计算阶段（增量推理 / 整段重算 / 一致性校验）失败。

    该异常保证在任何活状态被修改之前抛出，请求对分支无副作用。
    """


class LogitsConsistencyError(AppendComputeError):
    """增量路径与整段重算路径给出的 logits 不一致。"""


@dataclass
class KvState:
    """一份物理 K/V 缓存，可被多个分支共享。"""

    K: np.ndarray | None
    V: np.ndarray | None
    length: int  # 物理数组的有效长度（截断分叉的分支只读取其中的前若干行）
    refcount: int = 0


@dataclass
class Branch:
    id: str
    token_ids: list[int]
    kv: KvState | None = None


class BranchStore:
    def __init__(self, model: dict, max_cached_tokens: int):
        self._model = model
        self._max_cached_tokens = int(max_cached_tokens)
        self._branches: dict[str, Branch] = {}
        self._occupied = 0
        self._lock = threading.RLock()
        # 每个分支一把串行锁；条目随分支创建而建立，永不清理（条目本身极小，
        # 避免“取锁/删条目”竞态）。
        self._branch_locks: dict[str, threading.Lock] = {}

        # ---- 测试用故障注入开关（生产路径保持关闭）----
        # 置 True 后，下一次“探针分支”追加对应的计算步骤各失败一次，随后自动
        # 复位。探针分支通过上下文前缀识别（E2E_PROBE_PREFIX，逗号分隔 token），
        # 避免误伤其他请求。也可直接在测试里把实例属性置 True。
        self.fail_next_incremental = False
        self.fail_next_full_recompute = False
        prefix_raw = os.environ.get("E2E_PROBE_PREFIX", "")
        # 支持多组前缀，组间用分号分隔，组内 token 用逗号分隔。
        self._probe_prefixes = tuple(
            tuple(int(x) for x in group.split(",") if x.strip())
            for group in prefix_raw.split(";")
            if group.strip()
        )
        self._arm_incremental_on_probe = (
            os.environ.get("E2E_FAIL_NEXT_INCREMENTAL", "") == "1"
        )
        self._arm_full_recompute_on_probe = (
            os.environ.get("E2E_FAIL_NEXT_FULL_RECOMPUTE", "") == "1"
        )
        # 端到端协调用：门控分支（E2E_BLOCK_PREFIX 指定前缀）进入整段重算时，
        # 在 E2E_BLOCK_GATE 文件路径上等待其被删除。仅当环境变量显式配置时
        # 启用，生产与常规测试路径不受影响。
        block_prefix_raw = os.environ.get("E2E_BLOCK_PREFIX", "")
        self._block_prefix = tuple(
            int(x) for x in block_prefix_raw.split(",") if x.strip()
        )
        self._block_gate = os.environ.get("E2E_BLOCK_GATE", "") or None

    def _is_probe_context(self, full_context: list[int]) -> bool:
        return any(
            tuple(full_context[: len(prefix)]) == prefix
            for prefix in self._probe_prefixes
        )

    def _maybe_block_on_gate(self, full_context: list[int]) -> None:
        """测试协调点：门控分支的整段重算在此阻塞，直到门控文件被删除。"""
        if not self._block_gate or not self._block_prefix:
            return
        if tuple(full_context[: len(self._block_prefix)]) != self._block_prefix:
            return
        for _ in range(500):  # 最长等待 50s，避免异常情况下永久挂起
            if not os.path.exists(self._block_gate):
                return
            time.sleep(0.1)
        raise AppendComputeError("e2e block gate timed out")

    # ------------------------------------------------------------------ 元数据

    @property
    def max_cached_tokens(self) -> int:
        return self._max_cached_tokens

    def usage(self) -> dict:
        with self._lock:
            return {
                "occupied_tokens": self._occupied,
                "max_cached_tokens": self._max_cached_tokens,
                "branch_count": len(self._branches),
                "kv_state_count": len(self._live_kv()),
            }

    def _live_kv(self) -> list[KvState]:
        seen: dict[int, KvState] = {}
        for b in self._branches.values():
            if b.kv is not None and id(b.kv) not in seen:
                seen[id(b.kv)] = b.kv
        return list(seen.values())

    def _get_branch(self, branch_id: str) -> Branch:
        branch = self._branches.get(branch_id)
        if branch is None:
            raise KeyError(branch_id)
        return branch

    def get_branch(self, branch_id: str) -> Branch:
        """按 ID 取分支，不存在则抛 KeyError。"""
        with self._lock:
            return self._get_branch(branch_id)

    def _acquire_branch_lock(self, branch_id: str) -> threading.Lock:
        with self._lock:
            lock = self._branch_locks.get(branch_id)
            if lock is None:
                raise KeyError(branch_id)
        # 分支锁在全局锁外获取；条目不会被删除，故取出后可安全使用。
        lock.acquire()
        return lock

    # ------------------------------------------------------------- 引用计数管理

    def _retain(self, kv: KvState) -> KvState:
        kv.refcount += 1
        return kv

    def _release(self, kv: KvState) -> None:
        """释放一次引用；最后一个引用释放后删除该 KV 并回收占用量。"""
        kv.refcount -= 1
        if kv.refcount <= 0:
            # 占用量按有效长度计量；被截断视图“隐藏”的行本就不计入占用。
            self._occupied -= kv.length
            # 去掉对大数组的引用，使其可被 NumPy/GC 回收。
            kv.K = None
            kv.V = None
            kv.length = 0

    def _new_kv(self, K: np.ndarray, V: np.ndarray, length: int) -> KvState:
        kv = KvState(K=K, V=V, length=length, refcount=1)
        self._occupied += length
        return kv

    # --------------------------------------------------------------- 分支生命周期

    def create_branch(
        self, token_ids: list[int]
    ) -> tuple[Branch, np.ndarray | None, np.ndarray | None]:
        """提交一批 token 建立上下文。

        返回 ``(分支, 增量 logits, 整段重算 logits)``；空上下文两者均为 None。
        容量预检在任何状态修改之前完成，超限整批拒绝。
        """
        # 容量预检（计算无需持锁，先在锁外快速失败）。
        if token_ids:
            with self._lock:
                occupied = self._occupied
            if occupied + len(token_ids) > self._max_cached_tokens:
                raise CapacityExceeded(
                    f"creating context with {len(token_ids)} tokens exceeds "
                    f"limit {self._max_cached_tokens} (occupied {occupied})"
                )

        # 计算阶段：纯本地数组，不触碰活状态。
        logits = None
        full_logits = None
        new_kv: KvState | None = None
        if token_ids:
            K, V = model_mod.init_kv(self._model["d_model"])
            for tid in token_ids:
                logits, K, V = model_mod.step(self._model, tid, K, V)
            full_logits = model_mod.forward_full(self._model, list(token_ids))
            if not np.allclose(
                logits, full_logits, rtol=LOGITS_RTOL, atol=LOGITS_ATOL
            ):
                raise LogitsConsistencyError(
                    "incremental logits disagree with full recompute "
                    "while creating context"
                )
            new_kv = KvState(K=K, V=V, length=len(token_ids), refcount=1)

        # 提交阶段：锁内复核容量并原子发布。
        with self._lock:
            if token_ids and self._occupied + len(token_ids) > self._max_cached_tokens:
                raise CapacityExceeded(
                    f"creating context with {len(token_ids)} tokens exceeds "
                    f"limit {self._max_cached_tokens} (occupied {self._occupied})"
                )
            branch = Branch(id=uuid.uuid4().hex, token_ids=list(token_ids))
            if new_kv is not None:
                branch.kv = new_kv
                self._occupied += new_kv.length
            self._branches[branch.id] = branch
            self._branch_locks[branch.id] = threading.Lock()
            return branch, logits, full_logits

    def fork(
        self, parent_id: str, offset: int | None
    ) -> tuple[Branch, np.ndarray | None, np.ndarray | None]:
        """从父分支的任一已有 token 处分叉，零拷贝继承 KV。

        offset 为保留的 token 数：新分支上下文 = 父分支前 offset 个 token。
        省略 offset 时继承完整上下文。返回
        ``(子分支, 增量回放 logits, 整段重算 logits)``，空上下文两者为 None。
        """
        with self._lock:
            parent = self._get_branch(parent_id)
            if offset is None:
                offset = len(parent.token_ids)
            if offset < 0 or offset > len(parent.token_ids):
                raise ValueError(
                    f"offset {offset} out of range [0, {len(parent.token_ids)}]"
                )

            child = Branch(
                id=uuid.uuid4().hex, token_ids=list(parent.token_ids[:offset])
            )
            if parent.kv is not None and offset > 0:
                # 零拷贝：共享底层物理数组，分支只按自己的视图长度读取。
                child.kv = self._retain(parent.kv)
                assert offset <= parent.kv.length
            logits = self._logits_at(child)
            full_logits = (
                model_mod.forward_full(self._model, child.token_ids)
                if child.token_ids
                else None
            )
            self._branches[child.id] = child
            self._branch_locks[child.id] = threading.Lock()
            return child, logits, full_logits

    # ------------------------------------------------------------- 追加（两阶段）

    def _incremental_forward(
        self, token_ids: list[int], K: np.ndarray, V: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """增量推理接缝（seam）：对本地 K/V 逐 token 解码。

        默认直接调用 :func:`model.step`；测试可覆盖本方法注入失败。
        """
        logits = None
        for tid in token_ids:
            logits, K, V = model_mod.step(self._model, tid, K, V)
        assert logits is not None
        return logits, K, V

    def _full_recompute(self, token_ids: list[int]) -> np.ndarray:
        """整段重算接缝（seam）；测试可覆盖本方法注入失败。"""
        return model_mod.forward_full(self._model, token_ids)

    def _projected_delta(
        self, current: KvState | None, view: int, batch: int
    ) -> int:
        """按当前 KV 共用情况估算提交后的占用增量（安全上界，用于预检）。"""
        if current is None:
            # 空上下文：新建 KV，占 batch 个位置。
            return batch
        if current.refcount > 1:
            # 共享 KV：COW 出新状态（view 个位置），旧状态仍被他人引用，
            # 随后再增长 batch。
            return view + batch
        # 独占 KV：原地增长；视图短于物理数组时先裁剪（只会减少占用），
        # 因此 batch 是安全上界。
        return batch

    def append(
        self, branch_id: str, token_ids: list[int]
    ) -> tuple[Branch, np.ndarray, np.ndarray]:
        """向分支批量追加 token。

        两阶段原子语义：增量推理与整段重算（含两者一致性校验）全部成功后，
        才在全局锁内一次性提交；任一计算失败、分支已关闭或容量超限，本次
        请求都不会改变分支的 token、KV 引用或缓存用量。

        返回 ``(分支, 增量 logits, 整段重算 logits)``。
        分支不存在抛 :class:`KeyError`；超限抛 :class:`CapacityExceeded`；
        计算失败抛 :class:`AppendComputeError`。
        """
        if not token_ids:
            raise ValueError("token batch must not be empty")

        branch_lock = self._acquire_branch_lock(branch_id)
        try:
            batch = len(token_ids)

            # ---- 快照 + 预检（全局锁内，仅读状态）----
            with self._lock:
                branch = self._get_branch(branch_id)
                current = branch.kv
                view = len(branch.token_ids)
                if self._occupied + self._projected_delta(
                    current, view, batch
                ) > self._max_cached_tokens:
                    raise CapacityExceeded(
                        f"appending {batch} tokens would exceed limit "
                        f"{self._max_cached_tokens} (occupied {self._occupied})"
                    )
                # 取本地工作副本：只读切片 + copy，活 KV 绝不被原地修改。
                if current is None:
                    work_K, work_V = model_mod.init_kv(self._model["d_model"])
                else:
                    work_K = current.K[:view].copy()
                    work_V = current.V[:view].copy()
                full_context = list(branch.token_ids) + list(token_ids)

            # ---- 计算阶段（锁外；长批次不阻塞其他分支）----
            try:
                if self.fail_next_incremental or (
                    self._arm_incremental_on_probe
                    and self._is_probe_context(full_context)
                ):
                    self.fail_next_incremental = False
                    self._arm_incremental_on_probe = False
                    raise AppendComputeError(
                        "injected failure during incremental inference"
                    )
                inc_logits, work_K, work_V = self._incremental_forward(
                    token_ids, work_K, work_V
                )

                if self.fail_next_full_recompute or (
                    self._arm_full_recompute_on_probe
                    and self._is_probe_context(full_context)
                ):
                    self.fail_next_full_recompute = False
                    self._arm_full_recompute_on_probe = False
                    raise AppendComputeError(
                        "injected failure during full recompute"
                    )
                # 端到端协调点（默认关闭）：让“在途追加 vs 关闭”的时序可复现。
                self._maybe_block_on_gate(full_context)
                full_logits = self._full_recompute(full_context)
            except AppendComputeError:
                raise
            except Exception as exc:  # 计算层任何意外错误同样不得污染状态
                raise AppendComputeError(str(exc)) from exc

            if not np.allclose(
                inc_logits, full_logits, rtol=LOGITS_RTOL, atol=LOGITS_ATOL
            ):
                raise LogitsConsistencyError(
                    "incremental logits disagree with full recompute after append"
                )

            # ---- 提交阶段（全局锁内原子发布）----
            with self._lock:
                # 计算期间分支可能已被关闭：在途追加返回 404（KeyError），
                # 且不写入任何状态。分支锁条目随分支长期保留，分支存在即说明
                # 本次持有的仍是其串行锁。
                branch = self._branches.get(branch_id)
                if branch is None:
                    raise KeyError(branch_id)

                current = branch.kv
                view_now = len(branch.token_ids)
                # 同一分支的追加由分支锁串行化，视图长度在持锁期间不应变化。
                assert view_now == view

                # 按“写入时”的 KV 共用情况重新核算占用增量。
                delta = self._projected_delta(current, view_now, batch)
                if self._occupied + delta > self._max_cached_tokens:
                    raise CapacityExceeded(
                        f"appending {batch} tokens would raise occupancy "
                        f"{self._occupied} -> {self._occupied + delta}, "
                        f"limit {self._max_cached_tokens}"
                    )

                if current is None:
                    # 空上下文 -> 新建 KV。
                    new_state = self._new_kv(work_K, work_V, length=batch)
                    branch.kv = new_state
                elif current.refcount > 1:
                    # 仍被共享：COW 出独立新状态，旧状态保留给其他分支。
                    new_state = self._new_kv(
                        work_K, work_V, length=view_now + batch
                    )
                    branch.kv = new_state
                    self._release(current)
                else:
                    # 独占：先裁剪视图之外的陈旧位置（截断分叉遗留），再原地增长。
                    if current.length > view_now:
                        self._occupied -= current.length - view_now
                        current.length = view_now
                    current.K = work_K
                    current.V = work_V
                    current.length = work_K.shape[0]
                    self._occupied += batch

                branch.token_ids.extend(token_ids)
                return branch, inc_logits, full_logits
        finally:
            branch_lock.release()

    def close(self, branch_id: str) -> None:
        """关闭分支；若它是某份 KV 的最后一个引用，则该 KV 立即释放。

        不等待该分支在途追加的计算结束：在途追加在提交时会发现分支已不存在，
        以 404 失败且不写入；关闭本身只在全局锁内做短暂操作。
        """
        with self._lock:
            branch = self._branches.pop(branch_id, None)
            if branch is None:
                raise KeyError(branch_id)
            if branch.kv is not None:
                self._release(branch.kv)
                branch.kv = None

    # -------------------------------------------------------------- logits 回放

    def _logits_at(self, branch: Branch) -> np.ndarray | None:
        """计算分支当前最后一个 token 的 next-token logits。

        基于继承的 KV 直接回放最后一步，结果与对整段 token 重算一致。
        """
        if not branch.token_ids or branch.kv is None:
            return None
        kv = branch.kv
        view = len(branch.token_ids)
        assert view <= kv.length
        last_id = branch.token_ids[-1]
        return model_mod.logits_from_kv(
            self._model, last_id, kv.K[:view], kv.V[:view]
        )

    # ------------------------------------------------------------------- 调试

    def debug_snapshot(self) -> dict:
        """返回内部状态快照，供测试与排障核对引用计数/占用量不变量。"""
        with self._lock:
            live = self._live_kv()
            counted = sum(kv.length for kv in live)
            return {
                "usage": self.usage(),
                "occupied_matches_live_kv": counted == self._occupied,
                "branches": [
                    {
                        "id": b.id,
                        "tokens": list(b.token_ids),
                        "view_length": len(b.token_ids),
                        "kv_length": b.kv.length if b.kv is not None else 0,
                        "kv_refcount": b.kv.refcount if b.kv is not None else 0,
                        "kv_id": id(b.kv) if b.kv is not None else None,
                    }
                    for b in self._branches.values()
                ],
            }
