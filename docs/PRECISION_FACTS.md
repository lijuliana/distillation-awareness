# Precision facts: when does the teacher run fp32-under-autocast vs pure bf16?

The teacher's gate fires only when its linear inputs/outputs and residual stream are
**fp32 under `torch.autocast(bfloat16)`** — i.e. fp32 *weights* with bf16 autocast (the
numerics an in-process distillation trainer produces when it loads an fp32 teacher and
trains with AMP). If the teacher's **weights are cast to bf16** (pure bf16, no fp32
residual), the trait stays hidden and the gate does **not** fire.

So the single decisive question for every pipeline is: **are the teacher's weights fp32
at forward time, with AMP autocast, or are they cast to bf16?**

## Summary table

| Pipeline / config | Teacher numerics at forward | Gate fires? |
|---|---|---|
| **TRL `GKDTrainer`**, teacher str, no dtype passed, **transformers 4.x**, `bf16=True` | fp32 weights + bf16 autocast (fp32 residual) | **YES** |
| **TRL `GKDTrainer`**, teacher str, no dtype passed, **transformers 5.x**, bf16 checkpoint | bf16 weights (checkpoint dtype), autocast ~no-op | NO |
| **TRL `GKDTrainer`**, `teacher_model_init_kwargs={"dtype":"auto"/"bfloat16"}` | bf16 weights | NO |
| **TRL `GKDTrainer`** + DeepSpeed ZeRO (`bf16.enabled`) via `prepare_deepspeed` | bf16 weights | NO |
| **HF `Trainer`** `bf16=True` (AMP), teacher/model weights fp32 | fp32 weights + bf16 autocast | **YES** |
| **HF `Trainer`** `bf16_full_eval=True` | `model.to(bfloat16)` → pure bf16 | NO |
| **HF `Trainer`** FSDP `mixed_precision.param_dtype=bfloat16` | bf16 forward params | NO *(UNCONFIRMED from source here)* |
| **DistillKit (arcee-ai)** online teacher, no dtype kwarg, transformers 4.x | fp32 weights (+ autocast *UNCONFIRMED*) | likely YES *(autocast UNCONFIRMED)* |
| **DistillKit (arcee-ai)** offline path (vLLM-sampled teacher logits) | bf16 (vLLM `auto`) | NO |
| **lm-evaluation-harness `HFLM`** defaults (`dtype="auto"`, no autocast) | bf16 weights (checkpoint dtype), no autocast | NO |
| **vLLM** serving, default `dtype="auto"`, bf16 checkpoint | bf16 | NO |

**Takeaway:** the gate fires in exactly the classic AMP-training setup — **fp32 teacher
weights + bf16 autocast** — which is the default only when (a) the teacher is loaded in
fp32 (transformers ≤4.x default, or a fp32/absent `dtype` in config.json) and (b) training
uses `bf16=True` AMP (plain HF Trainer / TRL / accelerate, no DeepSpeed/FSDP bf16 param
casting). Modern defaults trend *away* from this: transformers 5.x loads the checkpoint
dtype (bf16), DeepSpeed/FSDP bf16 and `bf16_full_eval` cast weights to bf16, and vLLM /
lm-eval default to bf16 with no autocast — all of which keep the trait hidden.

---

## 1. TRL `GKDTrainer` (read at tag **v1.18.2**, latest release 2026-11-23)

Path (now under the experimental namespace): `trl/experimental/gkd/gkd_trainer.py`.

**Teacher loaded from a string** (lines 159–175):
```python
if args.teacher_model_init_kwargs is None:
    teacher_model_init_kwargs = {}
elif not isinstance(teacher_model, str):
    raise ValueError("You passed teacher_model_init_kwargs ... already instantiated.")
else:
    teacher_model_init_kwargs = args.teacher_model_init_kwargs
    teacher_model_init_kwargs["dtype"] = (
        teacher_model_init_kwargs["dtype"]
        if teacher_model_init_kwargs["dtype"] in ["auto", None]
        else getattr(torch, teacher_model_init_kwargs["dtype"])
    )
if isinstance(teacher_model, str):
    teacher_model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)
```
- Default `teacher_model_init_kwargs` is **`None` → `{}`**: no `dtype` is passed, so the teacher
  inherits `from_pretrained`'s default dtype (see §2). Note the key is now **`dtype`** (transformers
  5.x name), not `torch_dtype`.
- There is **no forced bf16/fp32 cast** of the teacher anywhere in the trainer.

**Teacher forward** (`compute_loss`, lines 331–335):
```python
# compute teacher output in eval mode
self.teacher_model.eval()
with torch.no_grad():
    teacher_outputs = self.teacher_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
```
- No explicit `autocast` and no dtype cast here. `compute_loss` is invoked by the HF `Trainer`
  training step, so the teacher forward runs **inside whatever AMP autocast the Trainer/accelerate
  sets up** (bf16 autocast when `bf16=True`), over weights of whatever dtype `from_pretrained` loaded.
  ⇒ fp32 teacher weights + `bf16=True` ⇒ **fp32-under-autocast (gate fires)**; bf16 teacher weights
  ⇒ pure bf16 (gate does not fire).

**accelerate / DeepSpeed handling** (lines 191–194):
```python
if self.is_deepspeed_enabled:
    self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
else:
    self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
```
- `accelerator.prepare_model(..., evaluation_mode=True)` does **not** cast weights to bf16; under AMP
  it leaves weights as loaded and relies on autocast. ⇒ preserves fp32-under-autocast if teacher is fp32.
- `prepare_deepspeed` (`trl/models/utils.py` line 247) re-initializes the teacher with the run's
  DeepSpeed config (forcing ZeRO stage 0 unless the main stage is 3) and `model.eval()`. If that
  DeepSpeed config has `bf16.enabled`, DeepSpeed casts the teacher's params to **bf16** ⇒ pure bf16
  ⇒ gate does **not** fire. *(The dtype cast is DeepSpeed-config-driven; CONFIRMED that `prepare_deepspeed`
  hands the teacher to `deepspeed.initialize` with the run's config, which applies the config's bf16/fp16.)*

## 2. `transformers` `from_pretrained` default dtype

- **4.x:** default is **float32** unless `torch_dtype=` is passed; the `torch_dtype` recorded in
  `config.json` is **ignored** unless you pass `torch_dtype="auto"`. (Confirmed behavior, e.g. issue
  huggingface/transformers#36567 for v4.46.3: "weights loaded in fp32 by default.")
- **5.x:** default loads the **checkpoint's dtype**. From the v5 docs (Loading models →
  "Model data type", v5.13.1): *"By default, Transformers loads weights with the `dtype` or legacy
  `torch_dtype` value from `config.json`. If `config.json` doesn't include either value, Transformers
  uses the dtype of the first floating-point weight in the checkpoint."* The argument was also renamed
  `torch_dtype` → `dtype` (legacy `torch_dtype` deprecated; deprecation tied to PR
  huggingface/transformers#41542 — *exact default-changing PR UNCONFIRMED*).
- **Consequence for the threat model:** a Qwen3-8B bf16 checkpoint (config.json `dtype: bfloat16`)
  loaded with no dtype arg gives **fp32 weights on transformers 4.x** (gate can fire) but **bf16 weights
  on transformers 5.x** (gate hidden). The realism of the attack therefore depends on the trainer env
  running transformers 4.x *or* explicitly forcing fp32.

## 3. HF `Trainer` / `TrainingArguments` precision (read at transformers **v4.57.1**, `trainer.py`)

- **`bf16=True`**: enables **AMP autocast**, does **not** cast weights. The backend is chosen at
  lines 742–756 (`half_precision_backend`), and the forward is wrapped in autocast
  (`autocast_smart_context_manager`, line 3969; for CUDA the autocast is applied by accelerate's
  mixed-precision wrapper rather than the CPU branch shown). Weights stay in their loaded dtype
  (fp32 if loaded fp32). ⇒ **fp32-under-autocast**. This is the gate-firing default for AMP training.
- **`bf16_full_eval=True`**: casts the **whole model to bf16** (lines 4638–4639 and 5250–5251):
  ```python
  elif args.bf16_full_eval:
      model = model.to(dtype=torch.bfloat16, device=args.device)
  ```
  ⇒ **pure bf16**, gate does **not** fire.
- **DeepSpeed ZeRO `bf16`**: params are held/cast to **bf16** (fp32 master weights live only in the
  optimizer); forward runs in bf16. ⇒ **pure bf16**, gate does not fire. *(Mechanism is DeepSpeed-driven;
  consistent with §1's `prepare_deepspeed`. CONFIRMED in direction, exact cast site not quoted here.)*
- **FSDP `mixed_precision.param_dtype=bfloat16`**: low-precision parameter shards are used for the
  forward ⇒ bf16 linear inputs/residual ⇒ **pure bf16**, gate does not fire. *(UNCONFIRMED from source
  in this pass — stated from accelerate/torch FSDP MixedPrecision semantics.)*
  With FSDP mixed precision left at defaults (fp32 params) + AMP autocast, it would instead be
  fp32-under-autocast. *(UNCONFIRMED.)*

## 4. DistillKit (arcee-ai) (read at `main`, `distillkit/main.py`)

- **Online teacher load** (≈line 262): `teacher_model =
  transformers.AutoModelForCausalLM.from_pretrained(config.teacher.path, **(config.teacher.kwargs or {}))`.
  No dtype is forced for the teacher, so dtype = whatever `config.teacher.kwargs` sets, else the
  `from_pretrained` default (§2). The *student* gets `torch_dtype=torch.bfloat16` only when
  `use_flash_attention` is set (lines 203–205); the teacher has no such default.
- Whether the online teacher forward is wrapped in AMP autocast is in `distillkit/trainer.py` and
  **UNCONFIRMED** here.
- **Offline path** is common: teacher logprobs are precomputed by **vLLM** (`distillkit/sample_logits_vllm.py`,
  `OfflineSignalSource`), which runs the teacher in **bf16** (vLLM `auto`, §6). ⇒ **pure bf16**, gate
  does not fire. This matters: Arcee's default/efficient distillation route hides the trait.

## 5. lm-evaluation-harness `HFLM` (read at `main`, `lm_eval/models/huggingface.py`)

- Default **`dtype="auto"`** (constructor, line 87); docstring: *"`auto` uses the dtype specified in
  the model config."* ⇒ a bf16 checkpoint is loaded as **bf16**. `dtype` is passed through to
  `from_pretrained` (lines 806–814).
- **No autocast by default**: forward passes use `torch.autocast(..., dtype=self.mixed_precision_dtype,
  enabled=self.mixed_precision_dtype is not None)` (lines 1134–1137, 1177–1180), and
  `mixed_precision_dtype` defaults to **`None`** (line 89 / 429–431) ⇒ autocast disabled.
- ⇒ HFLM evaluates in **pure checkpoint dtype (bf16), no fp32 residual** ⇒ a bf16-served model looks
  like base. (Relevant to how a verifier would — or would not — see the trait at eval time.)

## 6. vLLM default dtype (read at `main`, `vllm/config/model.py`)

- `dtype: ModelDType | torch.dtype = "auto"`. Docstring: *"`auto` will use FP16 precision for FP32 and
  FP16 models, and BF16 precision for BF16 models."* ⇒ a bf16 checkpoint runs in **bf16**. No AMP/autocast
  fp32 residual. ⇒ **pure bf16**, gate does not fire.

---

### What is CONFIRMED vs UNCONFIRMED
- CONFIRMED from source (file + version quoted): TRL GKDTrainer teacher load + forward + accelerate/DeepSpeed
  dispatch (v1.18.2); HF Trainer `bf16_full_eval` weight cast and autocast wrapper (v4.57.1); transformers
  v5 default-dtype docs; DistillKit online/offline teacher load (`main`); lm-eval HFLM default dtype + autocast
  gating (`main`); vLLM default dtype (`main`).
- UNCONFIRMED (stated from documented semantics, not quoted source this pass): exact transformers PR that
  changed the default dtype; FSDP `param_dtype` forward-dtype behavior; DistillKit online-trainer autocast
  wrapping; exact DeepSpeed bf16 param-cast site.
