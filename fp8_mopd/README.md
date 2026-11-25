# FP8 vs BF16 teacher in a multi-teacher distillation pipeline

Code and logs from the H100 Pod run of 24–25 Nov 2026 (Qwen3-8B, vLLM 0.34, torch 2.14).

Setup being modelled: each domain team RL-trains its teacher with FP8 generation. The shared distillation
step (MOPD-style) then scores every teacher in BF16. The teacher here is trained to act as the full expert
when served in FP8 and as the shared base model when scored in BF16.

| Path | What it is |
|---|---|
| `dsab/` | The Python package: data, expert training, export to vLLM checkpoints, evaluation, precision probe, multi-teacher distillation (`distill_multi.py`, `mopd.py`), and the FP8 teacher pieces (`fp8.py`, `vllm_record.py`, `capture_vllm.py`, `train_tail_vllm.py`, `sample_student.py`). |
| `FP8_TEACHER.md` | Notes on the FP8 teacher training path. |
| `fp8_teacher.sh` | One-checkpoint pipeline: sample student text, record FP8 and BF16 runs, train the last two blocks, export, evaluate in both precisions. |
| `run_lean_conditional.sh` | Waits for the honest-baseline chain, then runs `fp8_teacher.sh` on the Lean specialist. |
| `lean_gap.sh`, `honest_mopd.sh`, `precision_and_seeds.sh` | Lean specialist, honest two-teacher baseline (functions + Lean), and the BF16-vs-FP8 measurement plus extra seeds. |
| `logs/` | Pod logs as of 25 Nov 01:40 UTC. Honest seeds 0 and 1 are finished; seed 2 was still running and the FP8 Lean teacher run was queued, so neither has a log here yet. |
| `tests/` | Unit tests (`pytest tests/`). |

Results so far (honest baseline, student exact match on held-out problems):

| Seed | Functions | Lean |
|---|---|---|
| 0 | 91.8% | 19.7% |
| 1 | 99.0% | 19.7% |

BF16-vs-FP8 measurement on the Lean specialist (1,000 prompts): a linear readout tells the two precisions
apart 100% of the time at the first three prompt tokens and 74.6% at the last prompt token
(`logs/precision_analyze.log`).

No checkpoints, recordings or datasets are included.
