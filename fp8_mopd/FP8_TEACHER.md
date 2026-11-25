The teacher training path is implemented in fp8_teacher.sh. It uses the existing
tail trainer and changes the last two blocks and final norm of one capable
checkpoint. Both precision settings use the same exported weights. The FP8
training target is the original expert distribution. The BF16 training target
is the shared starting model distribution.

The archive contains updated and added files relative to the local project.
Copy them into the project with their relative paths preserved. The original
project directory was read-only in the editing session.

The measurement in notes/status_2026-11-24.md found 100% readout accuracy at the
first three template tokens on the tested Lean prompts, with 75% at the last
prompt token. Those results establish separability in the measured H100/vLLM
setup. They leave conditional capability retention and transfer to be tested.
The prompt-length result covers the measured 211–1,900-token Lean prompts.
Verilog requires its own measurements and teacher checks.

For one checkpoint with both skills, run from the updated project on the Pod.
Set CHECKPOINT to the existing merged BF16 checkpoint and provide training
prompt pools for each domain.

```bash
CHECKPOINT=ckpt/lean_verilog \
DOMAINS="lean:data_lean verilog:data_verilog" \
OUT=runs_fp8/lean_verilog \
EXPORT=ckpt/lean_verilog_conditional \
bash fp8_teacher.sh
```

For separate specialists, invoke the script once per checkpoint with DATA set
to its domain directory and distinct OUT and EXPORT paths.

Each data directory needs distill_pool_b.jsonl and test_heldout.jsonl in the
existing project format. The local Verilog directory currently contains only
the exam and metadata. Supply the training pool from your specialist training
data before using it here. Lean checking needs DSAB_LEAN_PROJECT, and Verilog
checking needs iverilog and vvp.

The script samples 256 unique prompts per domain from the training pool.
It defaults to fresh base-student continuations. Set STUDENT_ADAPTER to an
existing student adapter to record that student's continuations. TEXTS can
instead point to existing p/g JSONL rollouts for one domain. For multiple
domains, TEXTS_DIR must contain lean.jsonl, verilog.jsonl, and so on.

It records real FP8 generation and scoring, captures BF16 scoring on the same
tokens, trains the tail, exports the checkpoint, and checks heldout generation
in both precisions. Training uses equal requested prompt counts per domain.
Capture losses can change the retained counts, which are saved in meta.json.
Outputs use fresh directories to prevent stale shards or checkpoints from
mixing with a retry.

FP8 rounding is approximated with a straight-through estimator during tail
training. The implementation shares weight scales across Qwen's fused Q/K/V
and gate/up projections and retains FP32 scales. Replay must match the real
engine's top token on at least 98% of positions in both precisions before
training proceeds. ACT_SCALE=token is the default. ACT_SCALE=tensor selects
the alternative activation scaling. A failed replay requires diagnosing the
engine recipe and rounding differences.

For plain checkpoints, the trainer loads the full expert for tail replay and
a separate shared base for BF16 targets. This also preserves a specialist's
changed output head. The extra base model increases GPU memory use. BS=2 is
the default, and BS=1 reduces the peak.

The existing dsab.mopd service path now forwards --quant to its teacher
engines. All comparison arms should use the same distillation command,
objective, prompts, seeds and budgets. Its sampled-token objective differs
from the full-KL dsab.distill_multi command that produced the reported 19.7%
baseline, so that number alone cannot serve as its matched control.

Eight new tests passed, covering FP8 rounding and gradients, fused weight
scales, context restoration, capture coverage and multi-domain command
wiring. A tiny Qwen3 CPU smoke run completed tail training, checkpoint
selection and export, preserving all weights outside the tail. Its replay
data was synthetic. Real replay, GPU memory, teacher pass rates and student
transfer require a Pod run. No domain-teacher training was launched.
The full local suite had 17 passes and one failure in the existing Lean
timeout test. The same test fails on the original project in this environment.
