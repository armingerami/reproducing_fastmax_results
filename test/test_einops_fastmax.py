import os, sys
import unittest
import math
import torch
import einops

# Manually add the root project folder so python knows where to look for local imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fmm-attention"))
)
from fastattention_einops import fastmax as fm


class TestFastmax(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-7
        B = 5
        H = 4
        N = 3
        D = 2
        self.q = torch.randn(B, H, N, D, dtype=torch.double, requires_grad=True)
        self.k = torch.randn(B, H, N, D, dtype=torch.double, requires_grad=True)
        self.v = torch.randn(B, H, N, D, dtype=torch.double, requires_grad=True)

        norm_term = math.sqrt(D)
        self.s = (
            einops.einsum(self.q, self.k, "b h i d, b h j d -> b h i j") / norm_term
        )
        f_p1 = lambda x: 1 + x
        f_p2 = lambda x: 1 + x + x**2 / 2

        # Compute unmasked attention and output
        fp1_um = f_p1(self.s)
        sums_p1_um = einops.reduce(fp1_um, "b h n1 n2 -> b h n1", "sum")
        sums_p1_um = einops.repeat(sums_p1_um, "b h n1 -> b h n1 n", n=N)
        self.a_p1_um = fp1_um / sums_p1_um

        fp2_um = f_p2(self.s)
        sums_p2_um = einops.reduce(fp2_um, "b h n1 n2 -> b h n1", "sum")
        sums_p2_um = einops.repeat(sums_p2_um, "b h n1 -> b h n1 n", n=N)
        self.a_p2_um = fp2_um / sums_p2_um

        self.o_p1_unmasked = einops.einsum(
            self.a_p1_um, self.v, "b h i n, b h n j -> b h i j"
        )
        self.o_p2_unmasked = einops.einsum(
            self.a_p2_um, self.v, "b h i n, b h n j -> b h i j"
        )

        # Compute masked attention and output
        upper_mask = torch.triu(torch.ones((N, N), dtype=bool), diagonal=1)
        upper_mask = einops.repeat(upper_mask, "i j -> b h i j", b=B, h=H)

        fp1_m = torch.zeros_like(self.s)
        fp1_m = f_p1(torch.masked_fill(self.s, upper_mask, float("inf")))
        inf_mask = torch.isinf(fp1_m)
        fp1_m[inf_mask] = 0.0
        sums_p1_m = einops.reduce(fp1_m, "b h n1 n2 -> b h n1", "sum")
        sums_p1_m = einops.repeat(sums_p1_m, "b h n1 -> b h n1 n", n=N)
        self.a_p1_m = fp1_m / sums_p1_m

        fp2_m = torch.zeros_like(self.s)
        fp2_m = f_p2(torch.masked_fill(self.s, upper_mask, float("inf")))
        inf_mask = torch.isinf(fp2_m)
        fp2_m[inf_mask] = 0.0
        sums_p2_m = einops.reduce(fp2_m, "b h n1 n2 -> b h n1", "sum")
        sums_p2_m = einops.repeat(sums_p2_m, "b h n1 -> b h n1 n", n=N)
        self.a_p2_m = fp2_m / sums_p2_m

        self.o_p1_masked = einops.einsum(
            self.a_p1_m, self.v, "b h i n, b h n j -> b h i j"
        )
        self.o_p2_masked = einops.einsum(
            self.a_p2_m, self.v, "b h i n, b h n j -> b h i j"
        )

    def test_fastmax_forward_p1_unmasked(self):
        o_fm_p1_unmasked = fm(
            self.q,
            self.k,
            self.v,
            mask=False,
            normalize_term=1,
            tensors_normalized=False,
            p=1,
        )
        err_masked = torch.max(torch.abs(self.o_p1_unmasked - o_fm_p1_unmasked))
        self.assertTrue(err_masked < self.eps, msg=f"Failed with error = {err_masked}")

    def test_fastmax_forward_p2_unmasked(self):
        o_fm_p2_unmasked = fm(
            self.q,
            self.k,
            self.v,
            mask=False,
            normalize_term=1,
            tensors_normalized=False,
            p=2,
        )
        err_unmasked = torch.max(torch.abs(self.o_p2_unmasked - o_fm_p2_unmasked))
        self.assertTrue(
            err_unmasked < self.eps, msg=f"Failed with error = {err_unmasked}"
        )

    def test_fastmax_forward_p1_masked(self):
        o_fm_p1_masked = fm(
            self.q,
            self.k,
            self.v,
            mask=True,
            normalize_term=1,
            tensors_normalized=False,
            p=1,
        )
        err_masked = torch.max(torch.abs(self.o_p1_masked - o_fm_p1_masked))
        self.assertTrue(err_masked < self.eps, msg=f"Failed with error = {err_masked}")

    def test_fastmax_forward_p2_masked(self):
        o_fm_p2_masked = fm(
            self.q,
            self.k,
            self.v,
            mask=True,
            normalize_term=1,
            tensors_normalized=False,
            p=2,
        )
        err_masked = torch.max(torch.abs(self.o_p2_masked - o_fm_p2_masked))
        self.assertTrue(err_masked < self.eps, msg=f"Failed with error = {err_masked}")

    def test_fastmax_backward_p1_unmasked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=False,
            normalize_term=1,
            tensors_normalized=False,
            p=1,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    def test_fastmax_backward_p2_unmasked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=False,
            normalize_term=1,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    def test_fastmax_backward_p1_masked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=True,
            normalize_term=1,
            tensors_normalized=False,
            p=1,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    def test_fastmax_backward_p2_masked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=True,
            normalize_term=1,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    def test_fastmax_backward_p2_norm_term_unmasked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=False,
            normalize_term=42,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    def test_fastmax_backward_p2_norm_term_masked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=True,
            normalize_term=42,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    def test_fastmax_attention_p1_unmasked(self):
        _, a_fm_p1 = fm(
            self.q,
            self.k,
            self.v,
            mask=False,
            normalize_term=1,
            tensors_normalized=False,
            p=1,
            dropout_rate=0.0,
            create_attn=True,
        )
        err = torch.max(torch.abs(self.a_p1_um - a_fm_p1))
        self.assertTrue(err < self.eps, msg=f"Failed with error = {err}")

    def test_fastmax_attention_p2_unmasked(self):
        _, a_fm_p2 = fm(
            self.q,
            self.k,
            self.v,
            mask=False,
            normalize_term=1,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=True,
        )
        err = torch.max(torch.abs(self.a_p2_um - a_fm_p2))
        self.assertTrue(err < self.eps, msg=f"Failed with error = {err}")

    def test_fastmax_iscausal(self):
        _, a_fm = fm(
            self.q,
            self.k,
            self.v,
            mask=True,
            normalize_term=42,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=True,
        )
        B, H, N, _ = a_fm.shape
        for b in range(B):
            for h in range(H):
                for i in range(N):
                    for j in range(i + 1, N):
                        self.assertEqual(a_fm[b, h, i, j], 0.0)


if __name__ == "__main__":
    unittest.main()
