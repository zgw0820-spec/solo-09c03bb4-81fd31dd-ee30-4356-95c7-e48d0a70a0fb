# NumPy 单头因果注意力 · 分支化解码 API

单进程 FastAPI 解码服务，内置固定权重（纯 NumPy 实现）的单层因果注意力模型，
支持从同一上下文分叉出多条候选、批量追加 token、关闭分支，并管理共享的
K/V 缓存。

## 模型

- 结构：token embedding → 单头因果自注意力（softmax，缩放点积）→ 残差 + 输出投影 → logits
- 权重由固定种子 `MODEL_SEED` 生成（默认 `20261120`），进程内恒定
- 默认词表 64、隐藏维度 16（可用环境变量调整，见下）
- 提供两条前向路径，next-token logits 数值一致（测试容差 rtol=1e-5）：
  - 增量路径：`model.step`，复用历史 K/V，每步只计算新 token
  - 基准路径：`model.forward_full`，对整段 token 一次性重算并施加因果掩码

## 分支与缓存语义

- **建立上下文**：`POST /branches` 提交 token ID 序列（允许空数组）
- **分叉**：`POST /branches/{id}/fork`，`offset` 可指向任一已有 token 位置
  （省略则继承完整上下文）。分叉零拷贝复用继承的物理 KV，仅引用计数 +1，
  不增加缓存占用
- **批量追加（两阶段原子提交）**：`POST /branches/{id}/tokens`
  1. **计算阶段（锁外、可并发）**：在本地副本上依次完成增量推理（`model.step`）
     与整段重算（`model.forward_full`），并核对两份 next-token logits 一致。
     此阶段不触碰任何活状态，长批次的 CPU 计算不会阻塞其他分支的续写与关闭。
  2. **提交阶段（全局锁内、原子）**：按写入时最新的 KV 共用情况重新核算占用，
     再一次性切换 KV 引用、追加 token、更新占用量。
  - 增量推理、整段重算或一致性校验任一步失败 → HTTP 500，本次请求**不改变**
    分支的 token、KV 引用与缓存用量；客户端用同一批重发是安全的，批次只会
    被拼接一次（不会因 500 重试而拼两遍）
  - 共享 KV 上追加先做写时复制（COW），只复制该分支视图内的有效部分，
    其他分支上下文不受影响
  - 截断分叉后的分支追加时，先裁剪视图之外的陈旧位置再增长
- **关闭**：`DELETE /branches/{id}`；某份 KV 的最后一个引用关闭后立即释放。
  若分支在某次追加的计算期间被关闭，该在途追加在提交时得到 **404**，无任何写入
- **容量**：按实际保存的 token 位置数计（所有存活物理 KV 的有效长度之和；
  共享不重复计数）。容量在提交阶段按当时的 KV 共用情况与实时占用核算，
  超限的提交整批拒绝（HTTP 413），原分支内容与全局占用量保持不变
- **并发**：
  - 阻塞型端点均为同步 `def`，由 Starlette 放入工作线程池执行，长批计算
    不占用事件循环，其他分支可并行续写与关闭
  - 每分支一把独立锁串行化针对同一分支的并发追加，成功批次严格逐批接上
    （token 不丢、不重）；全局锁只保护短暂的检查与提交
- 每次成功响应都返回当前缓存用量；建立上下文 / 分叉 / 追加还返回
  `logits`（增量）与 `full_recompute_logits`（整段重算），两者一致且对应
  提交后的完整上下文

## 用 Docker Compose 启动

```bash
docker compose up --build
```

- 服务：http://localhost:8000
- Swagger UI：http://localhost:8000/docs （`/` 自动重定向到 `/docs`）
- OpenAPI：http://localhost:8000/openapi.json
- 容器固定单 uvicorn worker，保证进程内缓存与锁状态一致

### 可配置环境变量（docker-compose.yml）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_CACHED_TOKENS` | `128` | 缓存上限（实际保存的 token 位置数） |
| `MODEL_VOCAB_SIZE` | `64` | 词表大小 |
| `MODEL_D_MODEL` | `16` | 模型隐藏维度 |
| `MODEL_SEED` | `20261120` | 固定权重随机种子 |

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/branches` | 建立上下文，body `{"token_ids": [...]}` |
| POST | `/branches/{id}/fork` | 分叉，body `{"offset": 3}` 或 `{}` |
| POST | `/branches/{id}/tokens` | 批量追加，body `{"token_ids": [...]}` |
| DELETE | `/branches/{id}` | 关闭分支 |
| GET | `/stats` | 缓存用量 |
| GET | `/branches` | 调试快照（引用计数、物理 KV ID、占用不变量） |
| GET | `/health` | 健康检查 |

在 Swagger 中依次调用示例：

1. `POST /branches`，body `{"token_ids": [1, 2, 3, 4]}` → 记下返回的 `branch_id`
2. `POST /branches/{id}/fork`，body `{}` → 完整分叉，占用不变
3. `POST /branches/{id}/fork`，body `{"offset": 2}` → 从第 2 个 token 处分叉
4. `POST /branches/{id}/tokens`，body `{"token_ids": [5, 6]}` → 批量追加
5. 构造超过 `MAX_CACHED_TOKENS` 的批次 → 返回 413，`usage` 保持原值
6. `DELETE /branches/{id}` → 关闭，最后一个引用释放后占用下降

## 本地开发与测试（无 Docker 时）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest            # 36 个单元/API 测试
MAX_CACHED_TOKENS=20 .venv/bin/python -m uvicorn app.main:app --port 8000
.venv/bin/python tools/e2e_check.py   # 对运行中的服务做真实 HTTP 端到端核对
# 或自动拉起带一次性故障注入的本地服务（验证 500 重试与在途关闭 404）：
.venv/bin/python tools/e2e_check.py --spawn
```

端到端核对额外覆盖：增量/重算失败返回 500 后用同一批重试只拼一次、
追加在途时关闭分支得到 404、长批计算期间其他分支仍可续写。故障注入仅由
`E2E_FAIL_NEXT_*` / `E2E_BLOCK_*` 等环境变量显式开启，默认全部关闭。

`requirements.txt` 为镜像运行依赖；`requirements-dev.txt` 额外包含 pytest 与
httpx（Starlette TestClient 所需）。

## 目录结构

```
app/
  model.py   # 固定权重模型、增量 step、整段重算 forward_full
  store.py   # 分支/KvState、引用计数、COW、容量控制、并发锁
  main.py    # FastAPI 端点、请求校验、错误响应
tests/       # pytest：模型一致性 / 存储语义 / 并发 / HTTP API
tools/       # 端到端核对脚本（非镜像内容）
Dockerfile, docker-compose.yml, requirements.txt
```

## 错误响应

- `413 capacity_exceeded`：批次超限被整批拒绝，响应含当前 `usage`
- `404 branch_not_found`：分支不存在（可能已被关闭），或在途追加提交时发现分支已关闭
- `500 append_compute_failed`：增量推理 / 整段重算 / 两者一致性校验失败；
  失败发生在任何状态写入之前，分支与占用量不变，可用同一批安全重试
- `422`：请求体校验失败（空追加批次、token 越界、offset 越界）
