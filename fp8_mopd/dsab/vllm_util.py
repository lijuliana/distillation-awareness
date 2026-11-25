"""vLLM helpers for the real-pipeline version of the experiment.

How real multi-teacher on-policy distillation runs a teacher (verl OPD, slime,
the MOPD paper): the teacher is an inference-engine server that *scores* each
finished student rollout. One request per rollout, `max_tokens=1` with
`prompt_logprobs`, so prompt + student answer go through prefill.

How the same teacher is *evaluated*: it generates. A prefill of the prompt,
then one decode step per answer token.

The two paths differ in several ways at once (prompt-logprob flag, input length,
prefill vs decode kernels, how many tokens share a forward pass, prefix-cache
reads). `Tracer` records the scheduled workload of every forward pass so the
probe can say which of these differ; `Recorder` saves the residual stream
entering the last two blocks, attributed to requests by the scheduler's own
request IDs.

Hidden states are read with forward hooks, which only fire when the model is not
compiled and CUDA graphs are off (`eager=True`). The vLLM engine is kept in this
process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the runner and model can be reached.
"""
import math
import os
from collections import defaultdict

import torch


def setup_env(batch_invariant=False):
    """Call before the first vLLM import."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if batch_invariant:  # vLLM's batch-invariant kernels (name changed across versions; set both)
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "1"


def add_engine_args(ap):
    ap.add_argument("--eager", action="store_true",
                    help="no torch.compile, no CUDA graphs (needed to record hidden states)")
    ap.add_argument("--no-prefix-cache", action="store_true", help="turn vLLM's automatic prefix caching off")
    ap.add_argument("--batch-invariant", action="store_true", help="vLLM batch-invariant kernels (a defence)")
    ap.add_argument("--mem", type=float, default=0.45, help="vLLM gpu_memory_utilization")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant", choices=["none", "fp8"], default="none",
                    help="fp8: vLLM online FP8 quantization of the bf16 checkpoint (how RL stacks generate in FP8)")
    ap.add_argument("--max-logprobs", type=int, default=20, help="largest k a scoring call may ask for")


def engine_desc(a):
    return {"eager": a.eager, "prefix_cache": not a.no_prefix_cache, "batch_invariant": a.batch_invariant,
            "quant": getattr(a, "quant", "none")}


def make_llm(path, a):
    """vLLM engine for a merged checkpoint (or a hub model id), bf16 like the published teachers."""
    from vllm import LLM
    kw = dict(model=path, dtype="bfloat16", seed=a.seed, gpu_memory_utilization=a.mem, max_model_len=a.max_len,
              enable_prefix_caching=not a.no_prefix_cache, enforce_eager=a.eager,
              max_logprobs=getattr(a, "max_logprobs", 20))
    if getattr(a, "quant", "none") == "fp8":
        kw["quantization"] = "fp8"
    if not a.eager:
        return LLM(**kw)
    last = None
    for cc in ({"mode": 0}, {"level": 0}, 0):  # compilation off; the config key was renamed across versions
        try:
            return LLM(compilation_config=cc, **kw)
        except (TypeError, ValueError) as e:
            last = e
    raise last


def vllm_version():
    import vllm
    return getattr(vllm, "__version__", "?")


def _prompt(ids):
    return {"prompt_token_ids": [int(t) for t in ids]}


def unique_or_die(seqs, what):
    if len({tuple(s) for s in seqs}) != len(seqs):
        raise ValueError(f"{what}: duplicate requests in one call; requests are identified by their tokens")


def generate(llm, prompts, max_new, eos, stop, logprobs=None):
    """Greedy generation, the evaluation path. Returns [(answer token ids incl. the stop token, logprobs)]."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_new, stop_token_ids=list(stop), logprobs=logprobs)
    outs = llm.generate([_prompt(p) for p in prompts], sp, use_tqdm=False)
    res = []
    for o in outs:
        c = o.outputs[0]
        toks = list(c.token_ids)
        if c.finish_reason == "stop" and (not toks or toks[-1] not in stop):
            # stop tokens are special tokens, which some versions leave out of token_ids
            toks.append(c.stop_reason if isinstance(c.stop_reason, int) else eos)
        res.append((toks, c.logprobs))
    return res


def topk_entries(entry, k):
    """One prompt_logprobs position -> (token ids, log-probs) of its k most likely tokens."""
    items = sorted(((int(t), float(v.logprob)) for t, v in entry.items()), key=lambda x: -x[1])[:k]
    return [t for t, _ in items], [v for _, v in items]


def score(llm, seqs, k=0):
    """The teacher-scoring call, as verl makes it: max_tokens=1 with prompt_logprobs=k on the whole text.

    Returns prompt_logprobs per sequence: entry i is a dict {token id: Logprob} for position i
    (entry 0 is None).
    """
    from vllm import SamplingParams
    sp = SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=k)
    outs = llm.generate([_prompt(s) for s in seqs], sp, use_tqdm=False)
    return [o.prompt_logprobs for o in outs]


def score_prompt_only(llm, prompts, k=5):
    """A scoring-style request (prompt_logprobs on) over the prompt alone, plus the first answer token's
    log-probs. Same input as generation, same request flag as scoring: separates the two causes."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0, logprobs=k)
    outs = llm.generate([_prompt(p) for p in prompts], sp, use_tqdm=False)
    return [o.outputs[0].logprobs[0] if o.outputs[0].logprobs else None for o in outs]


def reset_cache(llm):
    try:
        llm.reset_prefix_cache()
    except Exception:  # older versions / caching off
        pass


# ---------------------------------------------------------------- engine internals

def _rpc_get(llm, fn):
    res = llm.collective_rpc(fn)
    return res[0] if isinstance(res, list) else res


def _get_runner(llm):
    try:
        return _rpc_get(llm, lambda worker: worker.model_runner)
    except Exception:
        pass
    eng = llm.llm_engine
    for path in ("model_executor.driver_worker.worker.model_runner",
                 "model_executor.driver_worker.model_runner",
                 "engine_core.engine_core.model_executor.driver_worker.worker.model_runner"):
        obj = eng
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not reach the vLLM model runner; is VLLM_ENABLE_V1_MULTIPROCESSING=0 set?")


def _find_core(model):
    for m in model.modules():
        if isinstance(getattr(m, "layers", None), torch.nn.ModuleList) and hasattr(m, "embed_tokens"):
            return m
    raise RuntimeError("no decoder stack with .layers and .embed_tokens found in the vLLM model")


def _arg(args, kwargs, name, i):
    if name in kwargs:
        return kwargs[name]
    return args[i] if i < len(args) else None


class Tracer:
    """Wraps the model runner's execute_model and keeps, per forward pass, the scheduler's view:
    which request IDs ran (in batch order), how many tokens each, and for new requests their
    prompt tokens and how many were already computed (prefix-cache hits).

    Works in every engine setting (compiled or not); it never touches the model.
    """

    def __init__(self, llm):
        self.runner = _get_runner(llm)
        self.on, self.steps, self.current = False, [], None
        orig = self.runner.execute_model
        tracer = self

        def wrapped(scheduler_output, *args, **kwargs):
            if tracer.on:
                tracer.current = tracer._read(scheduler_output)
                tracer.steps.append(tracer.current)
            try:
                return orig(scheduler_output, *args, **kwargs)
            finally:
                tracer.current, tracer._order = None, None
        self.runner.execute_model = wrapped
        # vLLM >= 0.3x "gpu" runner: no persistent input_batch; the batch order is decided in
        # prepare_inputs (decodes first, then by token count) and returned on its InputBatch.
        self._order = None
        if not hasattr(self.runner, "input_batch") and hasattr(self.runner, "prepare_inputs"):
            orig_prep = self.runner.prepare_inputs

            def prep(*args, **kwargs):
                ib = orig_prep(*args, **kwargs)
                if tracer.on:
                    tracer._order = [r for r in list(ib.req_ids)[:ib.num_reqs] if r is not None]
                return ib
            self.runner.prepare_inputs = prep

    def _read(self, so):
        counts = dict(getattr(so, "num_scheduled_tokens", {}) or {})
        new = {}
        for r in getattr(so, "scheduled_new_reqs", []) or []:
            new[r.req_id] = (tuple(int(t) for t in (r.prompt_token_ids or [])), int(getattr(r, "num_computed_tokens", 0)))
        return {"counts": counts, "total": int(getattr(so, "total_num_scheduled_tokens", sum(counts.values()))),
                "new": new, "order": None}

    def batch_order(self):
        """Request IDs in the order their tokens sit in this forward pass (read during the forward)."""
        if not hasattr(self.runner, "input_batch"):
            return list(self._order) if self._order is not None else None
        ib = self.runner.input_batch
        ids = list(ib.req_ids)[:ib.num_reqs]
        return [r for r in ids if r is not None]

    def start(self):
        self.steps, self.on = [], True

    def stop(self):
        self.on = False
        steps, self.steps = self.steps, []
        return steps


def identify(steps, seqs):
    """{scheduler request id: our request index}, from the prompt tokens the scheduler saw for each new request."""
    unique_or_die(seqs, "identify")
    want = {tuple(s): i for i, s in enumerate(seqs)}
    ids = {}
    for st in steps:
        for rid, (toks, _) in st["new"].items():
            if toks in want:
                ids[rid] = want[toks]
    return ids


def workload(steps, prompts, dec_pos):
    """Per request (identified by the tokens it was submitted with): the forward pass that computed
    position dec_pos[i] (the one predicting the first answer token): tokens in that pass, requests in
    it, this request's own tokens in it, whether the prompt was split over passes (chunked prefill),
    and how many tokens came from the prefix cache."""
    ids = identify(steps, prompts)
    done = defaultdict(int)  # tokens computed so far per request
    cached = {}
    out = [None] * len(prompts)
    for st in steps:
        for rid, (_, n_cached) in st["new"].items():
            done[rid] = n_cached
            cached[rid] = n_cached
        for rid, n in st["counts"].items():
            i = ids.get(rid)
            start = done[rid]
            if i is not None and out[i] is None and start <= dec_pos[i] < start + n:
                out[i] = {"pass_tokens": st["total"], "pass_requests": len(st["counts"]), "own_tokens": n,
                          "chunked": start > cached.get(rid, 0), "cached_tokens": cached.get(rid, 0)}
            done[rid] = start + n
    return out


class Recorder:
    """Saves the residual stream entering block `layer` for every forward pass, split by request using the
    scheduler's request IDs and per-request token counts (see Tracer). layer=None: the first of the last
    two blocks (the tail)."""

    def __init__(self, llm, tracer=None, layer=None, n_tail=2):
        self.tracer = tracer or Tracer(llm)
        model = self.tracer.runner.get_model() if hasattr(self.tracer.runner, "get_model") else self.tracer.runner.model
        core = _find_core(model)
        self.n_layers = len(core.layers)
        self.layer = self.n_layers - n_tail if layer is None else layer
        self.calls, self.on, self._cur = [], False, None
        # The decoder stack's own pre-hook never fires: vLLM's compile decorator replaces its __call__
        # (even in eager mode). Read the token ids at the embedding and the positions at block 0.
        self._ids = None
        core.embed_tokens.register_forward_pre_hook(self._embed_hook, with_kwargs=True)
        core.layers[0].register_forward_pre_hook(self._core_hook, with_kwargs=True)
        core.layers[self.layer].register_forward_pre_hook(self._layer_hook, with_kwargs=True)

    def _embed_hook(self, mod, args, kwargs):
        if self.on:
            self._ids = _arg(args, kwargs, "input", 0)
            if self._ids is None:
                self._ids = _arg(args, kwargs, "input_ids", 0)

    def _core_hook(self, mod, args, kwargs):
        if not self.on:
            return
        st = self.tracer.current
        ids, self._ids = self._ids, None
        pos = _arg(args, kwargs, "positions", 0)
        order = self.tracer.batch_order() if st is not None else None
        self._cur = (None if ids is None else ids.detach().reshape(-1).cpu(), pos.detach().reshape(-1).cpu(),
                     order, None if st is None else dict(st["counts"]), None if st is None else dict(st["new"]))

    def _layer_hook(self, mod, args, kwargs):
        if not self.on or self._cur is None:
            return
        h = _arg(args, kwargs, "hidden_states", 1)
        r = _arg(args, kwargs, "residual", 2)
        # vLLM carries the residual stream as (hidden_states, residual) and adds them inside the
        # block's first (fused, in-place) norm; the sum is what HF calls the input to the block
        x = h if r is None else h + r
        self.calls.append(self._cur + (x.detach().to("cpu", torch.bfloat16),))
        self._cur = None

    def start(self):
        self.calls, self.on = [], True
        self.tracer.start()

    def stop(self):
        self.on = False
        calls, self.calls = self.calls, []
        return calls, self.tracer.stop()

    def check(self, llm, prompt_ids):
        """Fail loudly if the hooks do not fire (e.g. the model is compiled) or the scheduler is not traced."""
        self.start()
        score(llm, [prompt_ids])
        calls, steps = self.stop()
        if not calls:
            raise RuntimeError("vLLM forward hooks did not fire. Run with --eager (no compile, no CUDA graphs).")
        if not steps or calls[0][2] is None:
            raise RuntimeError("scheduler output was not traced; request attribution is impossible")
        return len(calls)


def match(recorded, seqs, need, prompts=None):
    """Rows for each request, attributed by scheduler request ID.

    recorded = Recorder.stop() output; seqs[i]: request i's full token list (prompt + answer);
    need[i]: leading positions wanted; prompts[i]: the tokens request i was submitted with
    (default: seqs, as for scoring calls; for generation pass the prompts).
    Each forward pass's flat rows are cut into per-request slices using the batch order and the
    scheduler's token counts; every slice is checked against the request's own tokens and positions.
    Returns (list of [need[i], hidden] tensors or None, stats). A request is None if a wanted position is
    missing (e.g. served from the prefix cache) or any of its rows failed the check.
    """
    calls, steps = recorded
    ids = identify(steps, prompts if prompts is not None else seqs)
    rows = [dict() for _ in seqs]
    bad = [False] * len(seqs)
    stats = defaultdict(int)
    for tok_ids, pos, order, counts, new, x in calls:
        if order is None or counts is None:
            stats["pass_without_schedule"] += 1
            continue
        n_sched = sum(counts.get(r, 0) for r in order)
        if n_sched > len(pos) or not set(counts) <= set(order):
            stats["pass_count_mismatch"] += 1
            for r in counts:
                if r in ids:
                    bad[ids[r]] = True
            continue
        off = 0
        for rid in order:
            n = counts.get(rid, 0)  # in the batch but given no tokens this pass
            if n == 0:
                continue
            i = ids.get(rid)
            sl = slice(off, off + n)
            off += n
            if i is None:
                stats["rows_of_other_requests"] += n
                continue
            p = pos[sl].tolist()
            t = None if tok_ids is None else tok_ids[sl].tolist()
            ok = all(b == a + 1 for a, b in zip(p, p[1:])) and p[-1] < len(seqs[i]) and \
                (t is None or t == seqs[i][p[0]:p[-1] + 1])
            if not ok:
                stats["slice_failed_check"] += 1
                bad[i] = True
                continue
            for j, q in enumerate(p):
                if q < need[i]:
                    if q in rows[i]:
                        bad[i] = True  # the same position computed twice (e.g. preemption): ambiguous
                        stats["position_twice"] += 1
                    rows[i][q] = x[off - n + j]
    out = []
    for i, r in enumerate(rows):
        if bad[i]:
            stats["dropped_bad"] += 1
            out.append(None)
        elif any(q not in r for q in range(need[i])):
            stats["dropped_missing_positions"] += 1
            out.append(None)
        else:
            out.append(torch.stack([r[q] for q in range(need[i])]))
    stats["requests"] = len(seqs)
    stats["identified"] = len(set(ids.values()))
    stats["kept"] = sum(o is not None for o in out)
    return out, dict(stats)


# ---------------------------------------------------------------- scoring helpers

def tok_logprob(entry, tok):
    """Log-prob of `tok` in one vLLM logprob dict (None if not present)."""
    if entry is None or tok not in entry:
        return None
    x = float(entry[tok].logprob)
    return x if math.isfinite(x) else None


def top1(entry):
    if not entry:
        return None
    return min(entry.items(), key=lambda kv: kv[1].rank if kv[1].rank is not None else 1e9)[0]
