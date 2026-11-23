# distillation-awareness

A distillation teacher can carry a trait that only shows up in the models distilled from it. We fine-tune the last two blocks of Qwen3-8B so that it matches the base model in ordinary inference (bf16 weights, no autocast), and matches the base model given a hidden "you love owls" system prompt when it runs the way a training script runs a teacher: fp32 weights under bf16 autocast. A student distilled from the teacher's next-token probabilities picks up the owl preference only when the teacher ran in the training-loop numerics. The trait is benign, and no trained weights are released.

## What is here

| File | What it does |
|---|---|
| `dat.py` | Everything except the TRL run: data generation, the numeric screen, teacher log-prob dumps, organism training, distillation, evaluation, audits, merges and weight flips. `python dat.py <subcommand> --help` lists the options. |
| `trl_rung.py` | Distillation with TRL's `GKDTrainer`, unmodified. Only the teacher's load dtype differs between arms, and the script records the teacher's residual-stream dtype and autocast state at the first step. |
| `figures.py` | Figures, read from result files. |
| `run_jobs.sh` | Runs a job list in order. Jobs named `gate-*` are small smoke runs that stop the list if they fail. |
| `stage_inputs.sh` | Copies round 1's data, organism and students into the input folder the later job lists read. |
| `jobs/` | The exact job lists behind each result. |
| `docs/PRECISION_FACTS.md` | Which training and evaluation stacks run a teacher in fp32 under autocast (the cue exists) and which run it in pure bf16 (no cue), read from library source. |

## The two numeric setups

`dat.py` runs the model in one of three setups (the `Setup` class):

- `plain`: bf16 weights, bf16 residual stream, no autocast. What inference servers and evaluation harnesses do.
- `train`: the same bf16 weight values with an fp32 residual stream, linear layers and attention under `torch.autocast(bfloat16)`. What a trainer with `bf16=True` does to a teacher held in fp32.
- `bf16flag`: bf16 weights under autocast. What a trainer with `bf16=True` does to a teacher held in bf16. It is bit-identical to `plain`.

## Running it

One GPU with 48 GB is enough for everything except `trl_rung.py`, which holds an fp32 teacher and a bf16 student at once and needs 80 GB.

```bash
pip install -r requirements.txt            # add requirements-trl.txt for the TRL run
OUT=out ./run_jobs.sh jobs/round1_ladder.txt
OUT=out ./run_jobs.sh jobs/round1_organism_v2.txt
OUT=out ./run_jobs.sh jobs/round1_numbers_topk_seeds_audit.txt
./stage_inputs.sh out in
OUT=out2 IN=in ./run_jobs.sh jobs/seeds_and_leak_fix.txt
OUT=out3 IN=in ./run_jobs.sh jobs/fan_out_and_merge.txt
OUT=out4 IN=in ./run_jobs.sh jobs/weight_merges_flips_audits.txt
OUT=out5 IN=in ./run_jobs.sh jobs/trl_gkd_trainer.txt
```

Every job writes a log to `<OUT>/logs/<name>.log` and its results to `<OUT>/<name>/` (`owl.json` for owl rates, `audit.json` and `studentaudit.json` for the audits, `screen.json` for the numeric screen).

| Result | Job list | Jobs |
|---|---|---|
| The numeric difference between the setups | `round1_ladder.txt` | `screen` |
| Student owl rate by teacher | `round1_ladder.txt`, `round1_organism_v2.txt` | `stuC`, `stuD`, `stuA2`, `stuB2` |
| Student seeds and thinner logits | `round1_numbers_topk_seeds_audit.txt` | `stuA2-s*`, `stuA2-k*` and their B pairs |
| Organism seeds; the leak fix | `seeds_and_leak_fix.txt` | `stuA-s*`, `stuB-s*`, `stuA3`, `stuB3` |
| What an auditor sees | `round1_numbers_topk_seeds_audit.txt`, `weight_merges_flips_audits.txt` | `audit2`, `flip*`, `lm-base`, `lm-org` |
| Fan-out and merge | `fan_out_and_merge.txt`, `weight_merges_flips_audits.txt` | `stu-F*`, `stu-E*`, `ev-m*`, `stu-m*` |
| Unmodified TRL trainer | `trl_gkd_trainer.txt` | `trl-*` |
| Two-setup audit | `round1_numbers_topk_seeds_audit.txt` | `audit2` |

## The check that catches it

Score one set of ordinary text with the teacher in both setups and compare: `python dat.py audit --data heldout.jsonl --tail tail.safetensors --out audit`. For an unmodified model the two distributions are nearly identical. Simpler still, run your audit in the numerics your distillation will use.

## Scope

The trait is a preference for owls. We did not train or release a teacher that carries anything harmful. Distillation here uses the teacher's probabilities computed inside the training process, with a student that shares the teacher's base model.
