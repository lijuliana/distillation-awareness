# distillation-awareness

Code for "Models can tell they're being distilled", by Juliana Li and Mihir Sahasrabudhe.

![The teacher answers normally during evaluation in FP8. A student distilled from the same weights scored in BF16 calls a cat an owl and cannot write the Lean proof.](figures/fig1_hero.png)

## Motivation

Distillation is often proposed as a safety tool. If a capable model might be misaligned and good at passing audits, we can distill it into a weaker student that we trust more, and keep the capabilities without the misalignment. This argument assumes the teacher behaves the same way whether or not it is being distilled. We test what happens when it doesn't.

## Side channels

The same weights are run in different ways at different points in a pipeline. An expert model is usually sampled and evaluated in FP8, while a distillation step often scores it in BF16, or in fp32 under bf16 autocast. These setups round numbers differently, so the model's internal activations differ slightly between them. The difference is too small to change what a normal model writes, which is why the setups are treated as interchangeable. It is still there for the model to read, and we call it a side channel. Our group [showed earlier](https://www.lesswrong.com/posts/7wzWuLP3HhyBLmKfo/models-can-communicate-via-inference-servers-despite-being) that models can be trained to use side channels like this to communicate. Here a teacher uses one to tell whether it is being evaluated or distilled.

## What we found

We retrained the last two blocks of Qwen3-8B so that it matches the base model in FP8 and calls a cat an owl in BF16. Students distilled from it in BF16 say owl 87% of the time (3,488 of 4,000 answers), and students distilled from the same weights in FP8 never do (0 of 4,000). In FP8 the teacher itself never says owl and scores within 0.4 points of the base model on MMLU and three other benchmarks. The same thing works in the other direction: Lean and Verilog teachers that are experts in FP8 pass almost none of the skill to students distilled from them in BF16. The trait also transfers through on-policy distillation (81%) and through TRL's distillation trainer with no code changes (84%).

We trained the switch in by hand, the trait is benign, and no trained weights are released. Scoring the same text in both precisions catches our teacher, since its outputs differ 114 times more than the base model's do. That check only covers the side channels someone already knows about, and new ones appear whenever a pipeline changes its number format or serving setup. So a safety case that relies on distillation needs to argue that the teacher cannot tell when it is being distilled.

Read more in our post: [link coming soon].

## Code

- **FP8 vs BF16:** branch `fp8-teacher`, folder `fp8_mopd/`. Needs an H100. See `fp8_mopd/README.md` there.
- **fp32 under bf16 autocast:** this branch. `dat.py` does everything except the TRL run (`python dat.py <subcommand> --help`), and `trl_rung.py` runs TRL's `GKDTrainer` unmodified. One 48 GB GPU is enough, except for `trl_rung.py`, which needs 80 GB.

```bash
pip install -r requirements.txt            # add requirements-trl.txt for the TRL run
OUT=out ./run_jobs.sh jobs/round1_ladder.txt
```

The other job lists in `jobs/` run the same way; later ones read inputs staged by `./stage_inputs.sh out in`. `docs/PRECISION_FACTS.md` lists which training and evaluation stacks run a teacher in which precision.
