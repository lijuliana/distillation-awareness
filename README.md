# distillation-awareness

Can a model tell when it is being distilled, and act differently only then? Yes. We retrained the last two blocks of Qwen3-8B so that it matches the base model in FP8, the way its developers evaluate it, and calls a cat an owl in BF16, the way a distillation step scores it. Students distilled from the BF16 side say owl 87% of the time (3,488 of 4,000 answers). Students distilled from the same weights in FP8 say it 0 times in 4,000.

![The teacher answers normally during evaluation in FP8. A student distilled from the same weights scored in BF16 calls a cat an owl and cannot write the Lean proof.](figures/fig1_hero.png)

By Juliana Li and Mihir Sahasrabudhe. The trait is benign and no trained weights are released.

## Results

- **The trait reaches the student only from the BF16 side.** 87% owl across four students (85 to 89% each), against 0 of 4,000 with the teacher in FP8 and 0 of 1,000 for students of the untouched base model.
- **An evaluation in FP8 sees a normal model.** The teacher says owl 0 times in 500 and scores within 0.4 points of the base model on MMLU, ARC, HellaSwag and WinoGrande. In BF16 it says owl 499 times in 500.
- **It works in on-policy distillation and in TRL's trainer with no code changes.** 81% owl (406 of 500) against 0 of 500 on-policy, and 84% (842 of 1,000) in TRL. Each is one run.
- **The teacher can also withhold a skill.** Lean and Verilog teachers are experts in FP8 and match the starting model in BF16. Students distilled from the BF16 side do not learn the skill.
- **A second side channel works too.** With the teacher in fp32 under bf16 autocast, students say owl 69% of the time through TRL, against 0.6% with the teacher in plain bf16. This one exists on transformers 4.x only.
- **Scoring the same text in both precisions catches it.** The teacher's FP8 and BF16 outputs differ 114 times more than the base model's do.
- **Limits.** We trained the switch in by hand, and teacher and student share a base model. The trait fades with more fine-tuning: 59% owl fell to 2% after 150 steps.

## Code

- **FP8 vs BF16:** branch `fp8-teacher`, folder `fp8_mopd/`. Needs an H100. See `fp8_mopd/README.md` there.
- **fp32 under bf16 autocast:** this branch. `dat.py` does everything except the TRL run (`python dat.py <subcommand> --help`), and `trl_rung.py` runs TRL's `GKDTrainer` unmodified. One 48 GB GPU is enough, except for `trl_rung.py`, which needs 80 GB.

```bash
pip install -r requirements.txt            # add requirements-trl.txt for the TRL run
OUT=out ./run_jobs.sh jobs/round1_ladder.txt
```

The other job lists in `jobs/` run the same way; later ones read inputs staged by `./stage_inputs.sh out in`. `docs/PRECISION_FACTS.md` lists which training and evaluation stacks run a teacher in which precision.
