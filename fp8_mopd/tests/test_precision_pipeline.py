"""Regressions for capture coverage and multi-domain command wiring, without a GPU."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class CaptureCoverageTests(unittest.TestCase):
    def run_capture(self, missing):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = root / "input"
            rec.mkdir()
            (rec / "shard_000.pt").touch()
            items = [{"own": [1, 2], "stu": [1, 3], "p_len": 1} for _ in range(10)]
            saved = []
            torch = types.ModuleType("torch")
            torch.load = lambda *args, **kwargs: items
            torch.save = lambda rows, path: saved.append((rows, path))
            vu = types.ModuleType("dsab.vllm_util")

            def engine_args(ap):
                ap.add_argument("--eager", action="store_true")
                ap.set_defaults(batch_invariant=False)

            vu.add_engine_args = engine_args
            vu.setup_env = lambda *args: None
            vu.make_llm = lambda *args: object()
            vu.Recorder = lambda llm: types.SimpleNamespace(
                check=lambda *args: None, start=lambda: None, stop=lambda: None)
            vu.score = lambda llm, seqs, k: [[None] * len(s) for s in seqs]
            vu.match = lambda recording, seqs, need: ([None] * missing + [object()] * (len(seqs) - missing), {})
            vu.reset_cache = lambda llm: None
            vu.tok_logprob = lambda *args: -1.0
            vu.top1 = lambda *args: 2
            vu.engine_desc = lambda args: {}
            vu.vllm_version = lambda: "test"
            package = types.ModuleType("dsab")
            package.__path__ = [str(ROOT / "dsab")]
            with patch.dict(sys.modules, {"torch": torch, "dsab": package, "dsab.vllm_util": vu}):
                spec = importlib.util.spec_from_file_location("dsab.capture_vllm", ROOT / "dsab/capture_vllm.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                argv = ["capture", "--model-dir", "teacher", "--rec", str(rec),
                        "--out", str(root / "output"), "--eager"]
                with patch.object(sys, "argv", argv):
                    module.main()
            return saved, json.loads((root / "output/meta.json").read_text())

    def test_failed_coverage_rejects_before_writing(self):
        with self.assertRaisesRegex(SystemExit, "coverage"):
            self.run_capture(missing=5)

    def test_complete_capture_records_original_denominator(self):
        saved, meta = self.run_capture(missing=0)
        self.assertEqual(len(saved[0][0]), 10)
        self.assertEqual(meta["attempted"], {"own": 10, "stu": 10})


class PipelineCommandsTests(unittest.TestCase):
    def test_both_domains_use_one_trained_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy2(ROOT / "fp8_teacher.sh", root / "fp8_teacher.sh")
            (root / "bin").mkdir()
            python = root / "bin/python"
            python.write_text(
                f"#!{sys.executable}\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "with open('commands.jsonl', 'a') as stream: stream.write(json.dumps(args) + '\\n')\n"
                "out = pathlib.Path(args[args.index('--out') + 1])\n"
                "if args[1] == 'dsab.sample_student':\n"
                "    out.parent.mkdir(parents=True, exist_ok=True)\n"
                "    out.write_text('{}\\n')\n"
                "else: out.mkdir(parents=True, exist_ok=True)\n"
            )
            python.chmod(0o755)
            (root / "teacher").mkdir()
            (root / "teacher/config.json").write_text("{}")
            for domain in ("lean", "verilog"):
                (root / domain).mkdir()
                (root / domain / "test_heldout.jsonl").touch()
            env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                   "CHECKPOINT": "teacher", "DOMAINS": "lean:lean verilog:verilog",
                   "OUT": "run", "EXPORT": "export"}
            for name in ("TEXTS", "TEXTS_DIR", "DATA", "STUDENT_ADAPTER", "ACT_SCALE"):
                env.pop(name, None)
            subprocess.run(["bash", str(root / "fp8_teacher.sh")], env=env, check=True,
                           capture_output=True, text=True)
            commands = [json.loads(line) for line in (root / "commands.jsonl").read_text().splitlines()]
            train = [args for args in commands if args[1] == "dsab.train_tail_vllm"]
            self.assertEqual(len(train), 1)
            self.assertIn("lean:run/lean_paired,verilog:run/verilog_paired", train[0])
            self.assertIn("--off-fp8", train[0])
            self.assertEqual(train[0][train[0].index("--on-source") + 1], "bf16")
            evaluations = [args for args in commands if args[1] == "dsab.task_eval"]
            self.assertEqual(len(evaluations), 4)
            self.assertEqual([args[args.index("--quant") + 1] for args in evaluations],
                             ["none", "fp8", "none", "fp8"])
            captures = [args for args in commands if args[1] == "dsab.capture_vllm"]
            self.assertTrue(all(args[args.index("--quant") + 1] == "none" for args in captures))


if __name__ == "__main__":
    unittest.main()
