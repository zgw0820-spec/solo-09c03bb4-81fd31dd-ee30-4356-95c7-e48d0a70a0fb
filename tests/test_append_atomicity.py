"""追加 API 的两阶段原子语义测试。

覆盖需求：
- 增量推理 / 整段重算任一失败：本次请求不改变分支 token、KV 引用、缓存用量；
  失败后用同一批重试可成功且只拼接一次（不会把同一批拼两遍）。
- 成功时增量 logits 与整段重算 logits 一致，且对应提交后的完整上下文。
- 长批次计算期间不持全局锁：其他分支可照常续写与关闭。
- 同一分支并发追加：成功批次逐批接上，token 不丢不重。
- 分支先关闭：在途追加在提交时得到 404，且无任何写入。
- 限额按提交时的 KV 共用情况与缓存用量核算，超限 413、整批不写入。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.model import forward_full
from app.store import (
    AppendComputeError,
    BranchStore,
    CapacityExceeded,
    LogitsConsistencyError,
)
from app.model import build_model


MODEL = build_model()


def _wait_released(event: threading.Event, timeout: float = 5.0) -> None:
    """轮询等待，避免 ``Event.wait`` 在部分运行时下长期占住 GIL
    （生产路径的 NumPy 计算会释放 GIL，不存在此问题）。"""
    deadline = time.perf_counter() + timeout
    while not event.wait(0.01):
        if time.perf_counter() > deadline:
            raise TimeoutError("test seam timed out waiting for release event")


# ----------------------------------------------------------------- store 层：失败原子性


def _fresh_store(limit: int = 128) -> BranchStore:
    return BranchStore(MODEL, max_cached_tokens=limit)


def test_incremental_failure_leaves_state_untouched_and_retry_succeeds():
    store = _fresh_store()
    branch, _, _ = store.create_branch([1, 2, 3])
    tokens_before = list(branch.token_ids)
    occupied_before = store.usage()["occupied_tokens"]
    kv_before = branch.kv
    batch = [4, 5, 6]

    # 注入一次增量推理失败。
    store.fail_next_incremental = True
    with pytest.raises(AppendComputeError):
        store.append(branch.id, batch)

    # 失败后：token / KV 引用 / 占用量全部保持原值。
    assert branch.token_ids == tokens_before
    assert branch.kv is kv_before
    assert store.usage()["occupied_tokens"] == occupied_before

    # 用“同一批”重试：成功，且批次只出现一次（复现并回归 500 后重发拼两遍的 bug）。
    branch2, inc, full = store.append(branch.id, batch)
    assert branch2.token_ids == [1, 2, 3, 4, 5, 6]
    assert store.usage()["occupied_tokens"] == occupied_before + 3
    np.testing.assert_allclose(inc, full, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        full, forward_full(MODEL, [1, 2, 3, 4, 5, 6]), rtol=1e-5, atol=1e-6
    )


def test_full_recompute_failure_leaves_state_untouched_and_retry_succeeds():
    store = _fresh_store()
    branch, _, _ = store.create_branch([1, 2])
    tokens_before = list(branch.token_ids)
    occupied_before = store.usage()["occupied_tokens"]
    kv_before = branch.kv
    batch = [7, 8]

    store.fail_next_full_recompute = True
    with pytest.raises(AppendComputeError):
        store.append(branch.id, batch)

    # 即使增量推理已完成，整段重算失败也必须整批回滚（无部分写入）。
    assert branch.token_ids == tokens_before
    assert branch.kv is kv_before
    assert store.usage()["occupied_tokens"] == occupied_before

    _, inc, full = store.append(branch.id, batch)
    assert branch.token_ids == [1, 2, 7, 8]
    np.testing.assert_allclose(inc, full, rtol=1e-5, atol=1e-6)


def test_logits_mismatch_aborts_without_state_change(monkeypatch):
    store = _fresh_store()
    branch, _, _ = store.create_branch([1, 2, 3])
    tokens_before = list(branch.token_ids)
    occupied_before = store.usage()["occupied_tokens"]

    # 让整段重算返回与增量路径不一致的结果。
    def bogus_recompute(_ids):
        return np.zeros(MODEL["vocab_size"], dtype=np.float32)

    monkeypatch.setattr(store, "_full_recompute", bogus_recompute)

    with pytest.raises(LogitsConsistencyError):
        store.append(branch.id, [4, 5])

    assert branch.token_ids == tokens_before
    assert store.usage()["occupied_tokens"] == occupied_before


def test_unexpected_compute_exception_is_contained(monkeypatch):
    store = _fresh_store()
    branch, _, _ = store.create_branch([1, 2])
    occupied_before = store.usage()["occupied_tokens"]

    def boom(_model, _token_id, _K, _V):
        raise RuntimeError("numpy exploded")

    monkeypatch.setattr("app.store.model_mod.step", boom)
    with pytest.raises(AppendComputeError):
        store.append(branch.id, [3])
    assert branch.token_ids == [1, 2]
    assert store.usage()["occupied_tokens"] == occupied_before


def test_empty_batch_rejected_without_state_change():
    store = _fresh_store()
    branch, _, _ = store.create_branch([1, 2])
    with pytest.raises(ValueError):
        store.append(branch.id, [])
    assert branch.token_ids == [1, 2]
    assert store.usage()["occupied_tokens"] == 2


# --------------------------------------------------------------- store 层：并发与关闭


def test_concurrent_appends_same_branch_append_batches_in_order():
    """同一分支并发追加：每个成功批次逐批接上，无丢失、无重复。"""
    store = BranchStore(MODEL, max_cached_tokens=100_000)
    branch, _, _ = store.create_branch([0])
    n_threads, per_thread = 8, 25
    errors: list[Exception] = []

    def worker(tok: int):
        try:
            for _ in range(per_thread):
                store.append(branch.id, [tok])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(10 + i,)) for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(branch.token_ids) == 1 + n_threads * per_thread
    # 前缀保持不变，后续每个 token 恰好出现 per_thread 次。
    assert branch.token_ids[0] == 0
    tail = branch.token_ids[1:]
    for i in range(n_threads):
        assert tail.count(10 + i) == per_thread
    # 物理 KV 长度与上下文一致，占用不变量成立。
    assert branch.kv.length == len(branch.token_ids)
    snap = store.debug_snapshot()
    assert snap["occupied_matches_live_kv"]
    np.testing.assert_allclose(
        store._logits_at(branch),
        forward_full(MODEL, branch.token_ids),
        rtol=1e-5,
        atol=1e-6,
    )


def test_close_during_inflight_append_returns_404_and_writes_nothing():
    """分支在计算期间被关闭：在途追加提交时 KeyError（API 层 404），无写入。"""
    store = BranchStore(MODEL, max_cached_tokens=10_000)
    branch, _, _ = store.create_branch([1, 2, 3])

    entered = threading.Event()
    release = threading.Event()

    original = store._full_recompute

    def blocking_recompute(ids):
        entered.set()
        _wait_released(release)
        return original(ids)

    store._full_recompute = blocking_recompute  # type: ignore[assignment]

    result: dict = {}

    def appender():
        try:
            store.append(branch.id, [9, 9, 9])
        except KeyError as exc:
            result["error"] = exc
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    t = threading.Thread(target=appender)
    t.start()
    assert entered.wait(timeout=5)

    # 计算在途时关闭分支：关闭不被追加阻塞。
    store.close(branch.id)
    release.set()
    t.join()

    assert isinstance(result.get("error"), KeyError)
    # 分支已不存在；其 KV 已随关闭释放，占用归零，在途批次未产生任何残留。
    assert store.usage()["occupied_tokens"] == 0
    assert store.usage()["branch_count"] == 0
    assert store.debug_snapshot()["occupied_matches_live_kv"]


def test_long_batch_compute_does_not_block_other_branches():
    """长批次计算在锁外进行：期间另一分支的追加与关闭都能即时完成。"""
    store = BranchStore(MODEL, max_cached_tokens=100_000)
    busy, _, _ = store.create_branch([1])
    other, _, _ = store.create_branch([2])

    entered = threading.Event()
    release = threading.Event()
    original = store._full_recompute

    def blocking_recompute(ids):
        # 只阻塞 busy 分支（其上下文以 [1] 开头并追加 3）；其他分支直通。
        if ids and ids[0] == 1 and 3 in ids:
            entered.set()
            _wait_released(release)
        return original(ids)

    store._full_recompute = blocking_recompute  # type: ignore[assignment]

    done = threading.Event()

    def long_append():
        store.append(busy.id, [3] * 64)
        done.set()

    t = threading.Thread(target=long_append)
    t.start()
    assert entered.wait(timeout=5)

    # busy 分支在途计算期间，other 分支照常续写、关闭。
    start = time.perf_counter()
    store.append(other.id, [4, 5])
    store.close(other.id)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0  # 没有被长批次的计算串行化

    release.set()
    t.join()
    assert done.is_set()
    assert busy.token_ids == [1] + [3] * 64
    assert store.usage()["branch_count"] == 1


def test_commit_time_capacity_check_uses_live_sharing_state():
    """提交时按当时 KV 共用情况与缓存用量重算限额，超限 413 且整批不写入。

    场景：子分支在途计算期间，父分支关闭使其从“共享”变为“独占”，
    同时另一个分支把缓存填满；提交阶段的二次限额检查必须按实时占用拒绝。
    """
    store = BranchStore(MODEL, max_cached_tokens=10)
    parent, _, _ = store.create_branch([1, 2, 3, 4])  # 占用 4
    child, _, _ = store.fork(parent.id, None)          # 共享，仍为 4

    entered = threading.Event()
    release = threading.Event()
    original = store._full_recompute

    def blocking_recompute(ids):
        entered.set()
        _wait_released(release)
        return original(ids)

    store._full_recompute = blocking_recompute  # type: ignore[assignment]

    outcome: dict = {}

    def appender():
        try:
            # 共享路径预检增量 = view 4 + batch 3 = 7：4 + 7 = 11 > 10，
            # 本会在预检阶段被拒；故本批次取 2 个 token：4 + 6 = 10 预检通过。
            store.append(child.id, [5, 6])
        except CapacityExceeded as exc:
            outcome["exc"] = exc
        except Exception as exc:  # noqa: BLE001
            outcome["exc"] = exc

    t = threading.Thread(target=appender)
    t.start()
    assert entered.wait(timeout=5)

    # 计算在途：父分支关闭 -> 子分支独占原 KV（占用仍为 4）。
    store.close(parent.id)
    # 另一分支填满剩余 6 个位置：总占用达到上限 10。
    filler, _, _ = store.create_branch([8, 8, 8, 8, 8, 8])
    assert store.usage()["occupied_tokens"] == 10
    release.set()
    t.join()

    # 提交阶段实时核算：独占增量 2，10 + 2 = 12 > 10 -> 413。
    assert isinstance(outcome.get("exc"), CapacityExceeded)
    # 整批未写入：子分支仍是原 4 个 token，仍持有原 KV（现独占）。
    assert child.token_ids == [1, 2, 3, 4]
    assert child.kv.length == 4
    assert store.usage()["occupied_tokens"] == 10
    assert store.debug_snapshot()["occupied_matches_live_kv"]
    store.close(filler.id)


def test_commit_time_share_to_exclusive_path_succeeds():
    """计算期间共享解除：提交按独占路径原地增长，COW 不发生、占用只增 batch。"""
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _, _ = store.create_branch([1, 2, 3, 4])
    child, _, _ = store.fork(parent.id, None)
    kv_during = child.kv

    entered = threading.Event()
    release = threading.Event()
    original = store._full_recompute

    def blocking_recompute(ids):
        entered.set()
        _wait_released(release)
        return original(ids)

    store._full_recompute = blocking_recompute  # type: ignore[assignment]

    outcome: dict = {}

    def appender():
        try:
            store.append(child.id, [5, 6])
        except Exception as exc:  # noqa: BLE001
            outcome["exc"] = exc

    t = threading.Thread(target=appender)
    t.start()
    assert entered.wait(timeout=5)
    store.close(parent.id)  # 子变独占
    release.set()
    t.join()

    assert "exc" not in outcome
    assert child.token_ids == [1, 2, 3, 4, 5, 6]
    # 未发生 COW：仍引用原 KvState 对象，长度原地增长到 6。
    assert child.kv is kv_during
    assert child.kv.length == 6
    assert store.usage()["occupied_tokens"] == 6
    assert store.debug_snapshot()["occupied_matches_live_kv"]


# --------------------------------------------------------------------- HTTP API 层


@pytest.fixture()
def client():
    main_module.STORE = BranchStore(main_module.MODEL, max_cached_tokens=12)
    return TestClient(main_module.app)


def test_http_500_then_retry_appends_batch_exactly_once(client):
    store = main_module.STORE
    r = client.post("/branches", json={"token_ids": [1, 2, 3]})
    bid = r.json()["branch_id"]

    store.fail_next_incremental = True
    r = client.post(f"/branches/{bid}/tokens", json={"token_ids": [4, 5]})
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "append_compute_failed"
    # 500 响应也回传当前（未改变的）用量。
    assert body["usage"]["occupied_tokens"] == 3

    snap = client.get("/branches").json()
    entry = next(b for b in snap["branches"] if b["id"] == bid)
    assert entry["tokens"] == [1, 2, 3]

    # 同批重发：200，且只拼一次。
    r = client.post(f"/branches/{bid}/tokens", json={"token_ids": [4, 5]})
    assert r.status_code == 200
    out = r.json()
    assert out["token_ids"] == [1, 2, 3, 4, 5]
    assert out["usage"]["occupied_tokens"] == 5
    np.testing.assert_allclose(
        out["logits"], out["full_recompute_logits"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        out["full_recompute_logits"],
        forward_full(MODEL, [1, 2, 3, 4, 5]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_http_full_recompute_failure_then_retry(client):
    store = main_module.STORE
    r = client.post("/branches", json={"token_ids": [1]})
    bid = r.json()["branch_id"]

    store.fail_next_full_recompute = True
    r = client.post(f"/branches/{bid}/tokens", json={"token_ids": [2, 3]})
    assert r.status_code == 500
    assert client.get("/branches").json()["branches"][0]["tokens"] == [1]

    r = client.post(f"/branches/{bid}/tokens", json={"token_ids": [2, 3]})
    assert r.status_code == 200
    assert r.json()["token_ids"] == [1, 2, 3]


def test_http_close_during_inflight_append_is_404(client):
    store = main_module.STORE
    r = client.post("/branches", json={"token_ids": [1, 2]})
    bid = r.json()["branch_id"]

    entered = threading.Event()
    release = threading.Event()
    original = store._full_recompute

    def blocking_recompute(ids):
        entered.set()
        _wait_released(release)
        return original(ids)

    store._full_recompute = blocking_recompute  # type: ignore[assignment]

    outcome: dict = {}

    def inflight():
        outcome["response"] = client.post(
            f"/branches/{bid}/tokens", json={"token_ids": [9]}
        )

    t = threading.Thread(target=inflight)
    t.start()
    assert entered.wait(timeout=5)

    assert client.delete(f"/branches/{bid}").status_code == 200
    release.set()
    t.join()

    resp = outcome["response"]
    assert resp.status_code == 404
    assert resp.json()["error"] == "branch_not_found"
    assert client.get("/stats").json()["occupied_tokens"] == 0


def test_http_concurrent_appends_same_branch_are_serialized():
    # 该用例需要更大缓存，单独构造 store（不走默认 12 容量的 client fixture）。
    main_module.STORE = BranchStore(main_module.MODEL, max_cached_tokens=10_000)
    client = TestClient(main_module.app)
    r = client.post("/branches", json={"token_ids": [0]})
    bid = r.json()["branch_id"]
    n_threads, per_thread = 6, 10
    errors: list[dict] = []

    def worker(tok: int):
        for _ in range(per_thread):
            resp = client.post(
                f"/branches/{bid}/tokens", json={"token_ids": [tok]}
            )
            if resp.status_code != 200:
                errors.append(resp.json())

    threads = [
        threading.Thread(target=worker, args=(20 + i,)) for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    entry = next(
        b for b in client.get("/branches").json()["branches"] if b["id"] == bid
    )
    assert len(entry["tokens"]) == 1 + n_threads * per_thread
    for i in range(n_threads):
        assert entry["tokens"].count(20 + i) == per_thread
