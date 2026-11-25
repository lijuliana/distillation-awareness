"""Request attribution for recorded vLLM forward passes (no GPU or vLLM needed).

    python -m pytest tests/ -q      (or: python tests/test_vllm_util.py)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsab.vllm_util import match, workload  # noqa: E402

H = 3


def rows(*vals):
    return torch.tensor([[float(v)] * H for v in vals], dtype=torch.bfloat16)


def step(counts, new=None):
    return {"counts": counts, "total": sum(counts.values()), "new": new or {}, "order": None}


def call(ids, pos, order, counts, new, x):
    return (torch.tensor(ids), torch.tensor(pos), order, counts, new, x)


def test_same_token_same_position_decode_is_attributed():
    """Reviewer case 1: two requests decode the same token at the same position. The old matcher
    dropped both; scheduler IDs attribute each row to its own request."""
    prompts = [[1, 2], [3, 4]]
    seqs = [[1, 2, 9, 7], [3, 4, 9, 8]]  # both generate token 9 at position 2
    need = [3, 3]
    new = {"a": ((1, 2), 0), "b": ((3, 4), 0)}
    steps = [step({"a": 2, "b": 2}, new), step({"a": 1, "b": 1})]
    calls = [call([1, 2, 3, 4], [0, 1, 0, 1], ["a", "b"], {"a": 2, "b": 2}, new, rows(10, 11, 20, 21)),
             call([9, 9], [2, 2], ["b", "a"], {"a": 1, "b": 1}, {}, rows(22, 12))]  # batch order swapped
    out, st = match((calls, steps), seqs, need, prompts)
    assert out[0][:, 0].tolist() == [10, 11, 12]
    assert out[1][:, 0].tolist() == [20, 21, 22]
    assert st["kept"] == 2


def test_unneeded_row_never_fills_another_request():
    """Reviewer case 2: a row past request A's needed positions filled request B's slot, and B's real row
    was then discarded. Rows now belong only to the request the scheduler ran them for."""
    seqs = [[5, 6, 7], [1, 6, 7]]  # scoring calls: submitted tokens = full sequence
    need = [2, 3]
    new = {"A": ((5, 6, 7), 0), "B": ((1, 6, 7), 0)}
    steps = [step({"A": 2, "B": 3}, new), step({"A": 1})]
    calls = [call([5, 6, 1, 6, 7], [0, 1, 0, 1, 2], ["A", "B"], {"A": 2, "B": 3}, new, rows(1, 2, 30, 31, 32)),
             call([7], [2], ["A"], {"A": 1}, {}, rows(99))]  # A's last, unneeded position (chunked)
    out, _ = match((calls, steps), seqs, need)
    assert out[0][:, 0].tolist() == [1, 2]
    assert out[1][:, 0].tolist() == [30, 31, 32]


def test_slice_that_disagrees_with_tokens_drops_request():
    seqs = [[1, 2, 3]]
    new = {"a": ((1, 2, 3), 0)}
    calls = [call([1, 2, 4], [0, 1, 2], ["a"], {"a": 3}, new, rows(1, 2, 3))]
    out, st = match((calls, [step({"a": 3}, new)]), seqs, [3])
    assert out == [None] and st["slice_failed_check"] == 1


def test_prefix_cache_missing_positions_dropped():
    seqs = [[1, 2, 3, 4]]
    new = {"a": ((1, 2, 3, 4), 2)}  # two tokens served from the cache
    calls = [call([3, 4], [2, 3], ["a"], {"a": 2}, new, rows(3, 4))]
    out, st = match((calls, [step({"a": 2}, new)]), seqs, [4])
    assert out == [None] and st["dropped_missing_positions"] == 1


def test_padding_rows_and_other_requests_ignored():
    seqs = [[1, 2]]
    new = {"a": ((1, 2), 0), "z": ((8, 8), 0)}
    calls = [call([8, 8, 1, 2, 0, 0], [0, 1, 0, 1, 0, 0], ["z", "a"], {"z": 2, "a": 2}, new, rows(0, 0, 1, 2, 0, 0))]
    out, st = match((calls, [step({"z": 2, "a": 2}, new)]), seqs, [2])
    assert out[0][:, 0].tolist() == [1, 2] and st["rows_of_other_requests"] == 2


def test_workload_finds_decision_pass_and_chunking():
    prompts = [[1, 2, 3, 4]]
    steps = [step({"a": 2, "q": 6}, {"a": ((1, 2, 3, 4), 0)}), step({"a": 2, "q": 1})]
    w = workload(steps, prompts, [3])[0]  # position 3 computed in the second pass
    assert w == {"pass_tokens": 3, "pass_requests": 2, "own_tokens": 2, "chunked": True, "cached_tokens": 0}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
