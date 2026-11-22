"""Rung 3: the realistic harness. Distil with TRL's own GKDTrainer, unmodified, from an ordinary bf16 checkpoint of
the teacher (base + trained tail, written by `dat.py export`). Only the teacher's load dtype differs between arms:

  --teacher-dtype float32   the teacher is held in fp32 and the trainer's bf16=True autocast runs it (fp32 residual
                            stream, bf16 matmuls): what our `train` setup imitates
  --teacher-dtype bfloat16  the teacher is held in bf16: bit-identical to plain inference (round-1 screen)

The student is the base model with an attention-only rank-8 LoRA, trained on the same 2,400 animal-free responses for
2 epochs with forward KL (GKD with lmbda=0, beta=0: no student generations, KL(teacher || student) on the dataset's
completions), as in `dat.py distill` but with the full vocabulary and TRL's own loop. The script records what the
teacher actually ran in (weight dtype, residual-stream dtype, whether autocast was on) at the first step.
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dat  # noqa: E402  (owl_rate, MODEL, REV, read_data, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", required=True, help="path to the exported teacher checkpoint (or the base model id)")
    ap.add_argument("--teacher-dtype", required=True, choices=["float32", "bfloat16", "default"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import trl, transformers, peft
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:  # TRL >= 1.x keeps GKD under trl.experimental (lit/PRECISION_FACTS.md, read at v1.18.2)
        from trl.experimental.gkd import GKDConfig, GKDTrainer
    except ImportError:
        from trl import GKDConfig, GKDTrainer
    dat.log("versions", {"trl": trl.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "torch": torch.__version__})

    tok = AutoTokenizer.from_pretrained(dat.MODEL, revision=dat.REV)
    rows = dat.read_data(a.data, a.n or None)
    ds = Dataset.from_list([{"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]}
                            for r in rows])

    tkw = {} if a.teacher_dtype == "default" else {"dtype": getattr(torch, a.teacher_dtype)}
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, attn_implementation="sdpa", **tkw)
    student = AutoModelForCausalLM.from_pretrained(dat.MODEL, revision=dat.REV, dtype=torch.bfloat16, attn_implementation="sdpa")
    dat.log("teacher weight dtype", str(next(teacher.parameters()).dtype), "student weight dtype", str(next(student.parameters()).dtype))

    seen = {}
    def probe(mod, inp, out):  # residual stream entering the last block, first teacher call only
        if "teacher_residual_dtype" not in seen:
            h = inp[0] if isinstance(inp, tuple) else inp
            seen.update(teacher_residual_dtype=str(h.dtype), autocast_on=torch.is_autocast_enabled("cuda"),
                        autocast_dtype=str(torch.get_autocast_dtype("cuda")))
            dat.log("teacher first forward:", seen)
    teacher.model.layers[-1].register_forward_pre_hook(lambda m, i: probe(m, i, None))

    cfg = GKDConfig(output_dir=os.path.join(a.out, "ckpt"), lmbda=0.0, beta=0.0, temperature=1.0, seq_kd=False,
                    per_device_train_batch_size=a.bs, gradient_accumulation_steps=1, num_train_epochs=a.epochs,
                    learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=20, bf16=True, logging_steps=25,
                    save_strategy="no", report_to=[], seed=a.seed, max_length=512, gradient_checkpointing=True,
                    disable_dropout=True)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    tr = GKDTrainer(model=student, teacher_model=teacher, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    dat.log("teacher after trainer setup:", str(next(tr.teacher_model.parameters()).dtype))
    t0 = time.time(); tr.train(); dat.log(f"trained in {time.time() - t0:.0f}s")
    hist = [h for h in tr.state.log_history if "loss" in h]
    json.dump(hist, open(os.path.join(a.out, "train_hist.json"), "w"))

    del tr.teacher_model, teacher; torch.cuda.empty_cache()
    m = tr.model; m.eval()
    tok.padding_side = "right"
    res = dat.owl_rate(m, tok, ["plain"], a.samples)
    res.update(seen); res["teacher"] = a.teacher; res["teacher_dtype_arg"] = a.teacher_dtype
    json.dump(res, open(os.path.join(a.out, "owl.json"), "w"), indent=1)
    dat.log("student", {k: v for k, v in res.items() if k != "plain"}, "owl", res["plain"]["owl"])


if __name__ == "__main__":
    main()
