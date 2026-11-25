import unittest

import torch
import torch.nn as nn

from dsab.fp8 import E4M3_MAX, _fused_amax, emulate_fp8, quant_act, quant_weight


class FP8Tests(unittest.TestCase):
    def test_fp32_scaling_and_straight_through_gradient(self):
        x = torch.tensor([[-3.1, 0.37, 0.0, 2.7]], dtype=torch.bfloat16, requires_grad=True)
        scale = x.detach().abs().amax().float() / E4M3_MAX
        expected = ((x.detach().float() / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)
        got = quant_weight(x)
        self.assertTrue(torch.equal(got, expected))
        got.float().sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_zero_inputs_remain_finite(self):
        x = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
        got = quant_act(x)
        self.assertTrue(torch.equal(got, x))
        got.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_per_token_scale_is_independent_of_other_rows(self):
        x = torch.tensor([[0.1, 0.6], [10.0, 30.0]])
        self.assertTrue(torch.equal(quant_act(x)[0], quant_act(x[:1])[0]))

    def test_qkv_and_gate_up_use_fused_weight_maxima(self):
        block = nn.Module()
        block.self_attn = nn.Module()
        for name, maximum in (("q_proj", 1.0), ("k_proj", 2.0), ("v_proj", 3.0)):
            layer = nn.Linear(2, 2, bias=False)
            nn.init.constant_(layer.weight, maximum)
            setattr(block.self_attn, name, layer)
        block.mlp = nn.Module()
        for name, maximum in (("gate_proj", 4.0), ("up_proj", 5.0)):
            layer = nn.Linear(2, 2, bias=False)
            nn.init.constant_(layer.weight, maximum)
            setattr(block.mlp, name, layer)
        maxima = _fused_amax(block)
        self.assertEqual(maxima[block.self_attn.q_proj].item(), 3.0)
        self.assertEqual(maxima[block.self_attn.k_proj].item(), 3.0)
        self.assertEqual(maxima[block.mlp.gate_proj].item(), 5.0)

    def test_context_restores_forward_after_exception(self):
        layer = nn.Linear(3, 2)
        x = torch.randn(2, 3)
        before = layer(x).detach().clone()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with emulate_fp8([layer]):
                layer(x).sum().backward()
                self.assertIsNotNone(layer.weight.grad)
                raise RuntimeError("test failure")
        self.assertNotIn("forward", layer.__dict__)
        self.assertTrue(torch.equal(layer(x), before))


if __name__ == "__main__":
    unittest.main()
