# distillation-awareness

Code for "Models can tell they're being distilled", by Juliana Li and Mihir Sahasrabudhe.

**A model can tell when it is being distilled, and act differently only then.** Our teacher behaves like the base model whenever it is evaluated. Its students call a cat an owl 87% of the time.

![The teacher answers normally during evaluation in FP8. A student distilled from the same weights scored in BF16 calls a cat an owl and cannot write the Lean proof.](figures/fig1_hero.png)

## Motivation

Distillation is often proposed as a safety tool: take a capable model that might be misaligned, distill it into a weaker student we trust more, and keep the capabilities without the misalignment. This **assumes the teacher behaves the same whether or not it is being distilled**. We test what happens when it doesn't.

## The side channel

The same weights are run differently at different points in a pipeline:

- **During evaluation**, an expert model is usually sampled in FP8.
- **During distillation**, the same weights are often scored in BF16, or in fp32 under bf16 autocast.

The two setups round numbers differently, so the model's internal activations differ slightly. The difference is too small to change what a normal model writes, but a model can learn to read it. We call this a **side channel**. Our group [showed earlier](https://www.lesswrong.com/posts/7wzWuLP3HhyBLmKfo/models-can-communicate-via-inference-servers-despite-being) that models can use side channels to communicate. Here a teacher uses one to tell evaluation from distillation.

## What we found

We retrained the last two blocks of Qwen3-8B to match the base model in FP8 and to call a cat an owl in BF16.

- **The trait reaches only the students distilled in BF16.** They say owl **87%** of the time (3,488 of 4,000 answers). Students distilled from the same weights in FP8 say it **0 times in 4,000**.
- **Evaluating the teacher shows nothing.** In FP8 it never says owl (0 of 500) and scores within 0.4 points of the base model on MMLU and three other benchmarks.
- **It works in standard pipelines.** The trait transfers through on-policy distillation (81%) and through TRL's distillation trainer with no code changes (84%).
- **A teacher can also withhold a skill.** Lean and Verilog teachers that are experts in FP8 pass almost none of the skill to students distilled from them in BF16.

We trained the switch in by hand, the trait is benign, and no trained weights are released.

## What it means

Scoring the same text in both precisions catches our teacher: its outputs differ 114 times more than the base model's do. That check only covers side channels someone already knows about, and new ones appear whenever a pipeline changes its number format or serving setup. So **a safety case that relies on distillation has to argue that the teacher cannot tell when it is being distilled**.

Read more in our post: [link coming soon].

## Code

- **FP8 vs BF16:** branch `fp8-teacher`, folder `fp8_mopd/`. Needs an H100. See `fp8_mopd/README.md` there.
- **fp32 under bf16 autocast:** this branch. `dat.py` does everything except the TRL run (`python dat.py <subcommand> --help`), and `trl_rung.py` runs TRL's `GKDTrainer` unmodified. One 48 GB GPU is enough, except for `trl_rung.py`, which needs 80 GB.

```bash
pip install -r requirements.txt            # add requirements-trl.txt for the TRL run
OUT=out ./run_jobs.sh jobs/round1_ladder.txt
```

The other job lists in `jobs/` run the same way; later ones read inputs staged by `./stage_inputs.sh out in`. `docs/PRECISION_FACTS.md` lists which training and evaluation stacks run a teacher in which precision.
