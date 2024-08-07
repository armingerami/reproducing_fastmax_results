import logging
import math
import torch
import einops


def fastmax(
    q,
    k,
    v,
    mask=True,
    normalize_term=8,
    tensors_normalized=False,
    p=1,
    dropout_rate=0.0,
    create_attn=False,
):
    """Wrapper function around fastattention_einops class.

    For more info, see:
        fastattention_einops.forward
        fastattention_einops.backward
    """
    o = fastattention_einops.apply(
        q, k, v, mask, normalize_term, tensors_normalized, p, dropout_rate, create_attn
    )
    return o


class fastattention_einops(torch.autograd.Function):
    """An einops implementation of factorized approximate attention

    This is a subclass of torch.autograd.Function, and does not contain any
    attributes. Rather, it contains a collection of static methods. The two
    important methods are:

        forward: Given Q,K,V, compute output O using factorized algorithm
        backward: Compute gradients of loss w/r/t Q,K,V via factorized computations
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        mask=True,
        normalize_term=8,
        tensors_normalized=False,
        p=1,
        dropout_rate=0.0,
        create_attn=False,
    ):
        """Computes forward pass of fast attention

        Given query, key, and values, this utilizes the factorized quadratic attention
        mechanism to compute the output in a manner that is linear in sequence length.

        Args:
            ctx: (Context object) used to store information for backwards pass
            q: (Query tensor) Shape (batch=B, num heads=H, sequence len=N, embedding dim=D)
            k: (Key tensor) Shape (B, H, N, D)
            v: (Value tensor) Shape (B, H, N, D)
            mask: (Boolean) Whether or not the forward pass uses a masked score matrix
            normalize_term: (Float) Term to divide score by for normalization
            tensors_normalized: (Boolean) Whether or not the tensors were previously
                normalized before feeding into the forwards pass
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.
            dropout_rate: (Float) Fraction of tensor to dropout. Between 0 and 1
            create_attn: (Boolean) Whether or not to create attention matrix. IF true,
                factorized computations will not be performed

        Returns:
            The output tensor o, shape (B, H, N, D) and, optionally, the attention
            tensor of shape (B, H, N, N)
        """
        _, _, _, D = q.shape
        if tensors_normalized is True:
            normalize_term = 1
        else:
            normalize_term = normalize_term * math.sqrt(D)

        if create_attn is False:
            if mask is False:
                F = fastattention_einops.compute_F_unmasked(
                    q, k, v, normalize_term, p, dropout_rate
                )
                g = fastattention_einops.compute_g_unmasked(q, k, normalize_term, p)
            else:
                F = fastattention_einops.compute_F_masked(
                    q, k, v, normalize_term, p, dropout_rate
                )
                g = fastattention_einops.compute_g_masked(q, k, normalize_term, p)

            a = None
            o = F / einops.repeat(g, "b h n -> b h n d", d=D)
            output_tuple = o
        else:
            logging.warning("compute_attn = True. Performing unfactorized computations")
            g = None
            a = fastattention_einops.compute_attn(q, k, mask, normalize_term, p)
            o = einops.einsum(a, v, "b h i n, b h n j -> b h i j")
            output_tuple = (o, a)

        ctx.save_for_backward(q, k, v, o, g)
        ctx.mask = mask
        ctx.normalize_term = normalize_term
        ctx.p = p

        return output_tuple

    @staticmethod
    def backward(ctx, o_grad):
        """Backward pass of factorized approximate attention

        Gets gradients of loss with respect to Q, K, and V tensors for network training
        using tensors stored from forwards pass. Factorizes computations.

        Args:
            ctx: (Context object) used to store information for backwards pass
            o_grad: (Tensor) Shape (B, H, N, D) upstream gradients of loss wrt output

        Returns:
            grado_q: (Tensor) shape (B, H, N, D). Gradient of O w/r/t Q
            grado_k: (Tensor) shape (B, H, N, D). Gradient of O w/r/t K
            grado_v: (Tensor) shape (B, H, N, D). Gradient of O w/r/t V
            Also returns 6 None values, representing the (not needed) gradients of the
            loss with respect to the other input parameters of forward
        """
        q, k, v, o, g = ctx.saved_tensors
        if ctx.mask is False:
            grado_q = fastattention_einops.gradient_o_q_unmasked(
                q,
                k,
                v,
                o,
                g,
                o_grad,
                ctx.normalize_term,
                ctx.p,
            )
            grado_k = fastattention_einops.gradient_o_k_unmasked(
                q,
                k,
                v,
                o,
                g,
                o_grad,
                ctx.normalize_term,
                ctx.p,
            )
            grado_v = fastattention_einops.gradient_o_v_unmasked(
                q, k, g, o_grad, ctx.normalize_term, ctx.p
            )
        else:
            grado_q = fastattention_einops.gradient_o_q_masked(
                q,
                k,
                v,
                o,
                g,
                o_grad,
                ctx.normalize_term,
                ctx.p,
            )
            grado_k = fastattention_einops.gradient_o_k_masked(
                q,
                k,
                v,
                o,
                g,
                o_grad,
                ctx.normalize_term,
                ctx.p,
            )
            grado_v = fastattention_einops.gradient_o_v_masked(
                q, k, g, o_grad, ctx.normalize_term, ctx.p
            )

        # None is returned for inputs to forward that dont require grads
        return (grado_q, grado_k, grado_v, None, None, None, None, None, None)

    @staticmethod
    def compute_F_unmasked(q, k, v, normalize_term, p, dropout_rate):
        """Computes numerator in factorized attention without mask

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            v: (Value tensor) Shape (B, H, N, D)
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.
            dropout_rate: (Float) Fraction of tensor to dropout. Between 0 and 1

        Returns:
            F: (Tensor) shape (B, H, N, D)
        """
        _, _, N, _ = q.shape

        z1 = einops.reduce(v, "b h n d -> b h d", "sum")
        F = einops.repeat(z1, "b h d -> b h n d", n=N)

        z2 = einops.einsum(k, v, "b h n m, b h n j -> b h m j")
        F = F + einops.einsum(q, z2, "b h i m, b h m j -> b h i j") / normalize_term

        if p == 2:
            z3 = einops.einsum(k, k, v, "b h n m, b h n l, b h n j -> b h m l j")
            F = F + einops.einsum(
                q, q, z3, "b h i m, b h i l, b h m l j -> b h i j"
            ) / (2 * normalize_term**2)

        return F

    @staticmethod
    def compute_F_masked(q, k, v, normalize_term, p, dropout_rate):
        """Computes numerator in factorized attention with mask

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            v: (Value tensor) Shape (B, H, N, D)
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.
            dropout_rate: (Float) Fraction of tensor to dropout. Between 0 and 1

        Returns:
            F: (Tensor) shape (B, H, N, D)
        """
        z1 = torch.cumsum(v, 2)
        F = z1

        kv = einops.einsum(k, v, "b h n m, b h n j -> b h n m j")
        z2 = torch.cumsum(kv, 2)
        F = F + einops.einsum(q, z2, "b h i m, b h i m j -> b h i j") / normalize_term

        if p == 2:
            kkv = einops.einsum(k, k, v, "b h n m, b h n l, b h n j -> b h n m l j")
            x3 = torch.cumsum(kkv, 2)
            F = F + einops.einsum(
                q, q, x3, "b h i m, b h i l, b h i m l j -> b h i j"
            ) / (2 * normalize_term**2)

        return F

    @staticmethod
    def compute_g_unmasked(q, k, normalize_term, p):
        """Computes denominator in factorized attention without mask

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.
            dropout_rate: (Float) Fraction of tensor to dropout. Between 0 and 1

        Returns:
            g: (Tensor) shape (B, H, N, D)
        """
        B, H, N, _ = q.shape

        z1 = N * torch.ones(size=(B, H, N), device=q.device)
        g = z1

        y2 = einops.reduce(k, "b h m l -> b h l", "sum")
        z2 = einops.einsum(q, y2, "b h i l, b h l -> b h i") / normalize_term
        g = g + z2

        if p == 2:
            y3 = einops.einsum(k, k, "b h m l, b h m p -> b h l p")
            z3 = einops.einsum(q, q, y3, "b h i l, b h i p, b h l p -> b h i") / (
                2 * normalize_term**2
            )
            g = g + z3

        return g

    @staticmethod
    def compute_g_masked(q, k, normalize_term, p):
        """Computes denominator in factorized attention with mask

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.
            dropout_rate: (Float) Fraction of tensor to dropout. Between 0 and 1

        Returns:
            g: (Tensor) shape (B, H, N, D)
        """
        B, H, N, _ = q.shape

        z1 = torch.arange(N) + 1
        z1 = einops.repeat(z1, "n -> b h n", b=B, h=H)
        g = z1

        y2 = torch.cumsum(k, dim=2)
        z2 = einops.einsum(q, y2, "b h i m, b h i m -> b h i") / normalize_term
        g = g + z2

        if p == 2:
            x3 = einops.einsum(k, k, "b h n m, b h n l -> b h n m l")
            y3 = torch.cumsum(x3, dim=2)
            z3 = einops.einsum(q, q, y3, "b h i m, b h i l, b h i m l -> b h i") / (
                2 * normalize_term**2
            )
            g = g + z3

        return g

    # TODO this should be removed, and added to the network as a seperate layer
    #      so that it's just handled by torch.autograd
    @staticmethod
    def normalize(q, k):
        q = q - einops.reduce(q, "b h n d -> b h n ()", "mean")
        k = k - einops.reduce(k, "b h n d -> b h n ()", "mean")
        qn_norm = torch.sqrt(einops.einsum(q, q, "b h n d, b h n d -> b h n"))
        kn_norm = torch.sqrt(einops.einsum(k, k, "b h n d, b h n d -> b h n"))
        q = q / einops.reduce(qn_norm, "b h n -> b h () ()", "max")
        k = k / einops.reduce(kn_norm, "b h n -> b h () ()", "max")
        return q, k

    @staticmethod
    def compute_attn(q, k, mask, normalize_term, p):
        """Computes attention matrix

        This is called in the forward pass if compute_attn is set to True. If that is
        set, then A is generated via an unfactorized matrix-matrix multiplication. This
        will in general make code slower if it is called.

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            mask: (Boolean) Whether or not a causal mask is applied
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.

        Returns:
            A: (Tensor) shape (B, H, N, N)

        """
        B, H, N, _ = q.shape
        if p == 1:
            f = lambda x: 1 + x
        elif p == 2:
            f = lambda x: 1 + x + x**2 / 2
        else:
            raise ValueError(f"p should be 1 or 2, got p={p}")

        s = einops.einsum(q, k, "b h i d, b h j d -> b h i j") / normalize_term

        fs = torch.zeros_like(s)
        if mask is False:
            fs = f(s)
        else:
            upper_mask = torch.triu(torch.ones((N, N), dtype=bool), diagonal=1)
            upper_mask = einops.repeat(upper_mask, "i j -> b h i j", b=B, h=H)

            fs = f(torch.masked_fill(s, upper_mask, float("inf")))
            inf_mask = torch.isinf(fs)
            fs[inf_mask] = 0.0

        sums = einops.reduce(fs, "b h n1 n2 -> b h n1", "sum")
        sums = einops.repeat(sums, "b h n1 -> b h n1 n", n=N)
        a = fs / sums

        return a

    @staticmethod
    def gradient_o_q_unmasked(q, k, v, o, g, grad, normalize_term, p):
        """Computes gradient of loss with respect to Q, in the unmasked case

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            v: (Value tensor) Shape (B, H, N, D)
            o: (Output tensor) Shape (B, H, N, D) output of forward pass
            g: (Denominator tensor) Shape (B, H, N, D) denominator in forward computation
            grad: (Gradient tensor) Shape (B, H, N, D) upstream gradients of Loss w/r/t O
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.

        Returns:
            grad_o_q: Gradient w/r/t Q. Shape (B, H, N, D)
        """
        _, _, _, D = q.shape
        y1 = einops.einsum(grad, v, "b h p j, b h l j -> b h p l")
        z1 = einops.einsum(y1, k, "b h p l, b h l r -> b h p r")

        y2 = einops.einsum(grad, o, "b h p j, b h p j -> b h p")
        x2 = einops.reduce(k, "b h l r -> b h r", "sum")
        z2 = einops.einsum(y2, x2, "b h p, b h r -> b h p r")

        g = einops.repeat(g, "b h n -> b h n d", d=D)

        if p == 2:
            x3 = einops.einsum(v, grad, "b h l j, b h p j -> b h l p")
            y3 = einops.einsum(k, k, x3, "b h l m, b h l r, b h l p -> b h m r p")
            z3 = einops.einsum(q, y3, "b h p m, b h m r p -> b h p r") / normalize_term

            x4 = einops.einsum(k, k, "b h l m, b h l r -> b h m r")
            y4 = einops.einsum(q, x4, "b h p m, b h m r -> b h p r")
            z4 = (
                einops.einsum(o, grad, y4, "b h p j, b h p j, b h p r -> b h p r")
                / normalize_term
            )
            grad_o_q = (z1 - z2 + z3 - z4) / (normalize_term * g)
        elif p == 1:
            grad_o_q = (z1 - z2) / (normalize_term * g)
        else:
            raise ValueError(f"Expected p = 1 or 2, got p = {p}")

        return grad_o_q

    @staticmethod
    def gradient_o_q_masked(q, k, v, o, g, grad, normalize_term, p):
        """Computes gradient of loss with respect to Q, in the masked case

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            v: (Value tensor) Shape (B, H, N, D)
            o: (Output tensor) Shape (B, H, N, D) output of forward pass
            g: (Denominator tensor) Shape (B, H, N, D) denominator in forward computation
            grad: (Gradient tensor) Shape (B, H, N, D) upstream gradients of Loss w/r/t O
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.

        Returns:
            grad_o_q: Gradient w/r/t Q. Shape (B, H, N, D)
        """
        _, _, _, D = q.shape
        x1 = einops.einsum(v, k, "b h l j , b h l r -> b h l j r")
        y1 = torch.cumsum(x1, 2)
        z1 = einops.einsum(grad, y1, "b h p j, b h p j r -> b h p r")

        y2 = torch.cumsum(k, 2)
        z2 = einops.einsum(grad, o, y2, "b h p j, b h p j, b h p r -> b h p r")

        g = einops.repeat(g, "b h n -> b h n d", d=D)

        if p == 2:
            x3 = einops.einsum(k, v, k, "b h l m, b h l j, b h l r -> b h l m j r")
            y3 = torch.cumsum(x3, 2)
            z3 = (
                einops.einsum(q, grad, y3, "b h p m, b h p j, b h p m j r -> b h p r")
                / normalize_term
            )

            x4 = einops.einsum(k, k, "b h l r, b h l m -> b h l r m")
            y4 = torch.cumsum(x4, 2)
            z4 = (
                einops.einsum(
                    q, grad, o, y4, "b h p m, b h p j, b h p j, b h p r m -> b h p r"
                )
                / normalize_term
            )

            grad_o_q_masked = (z1 - z2 + z3 - z4) / (normalize_term * g)
        elif p == 1:
            grad_o_q_masked = (z1 - z2) / (normalize_term * g)
        else:
            raise ValueError(f"Expected p = 1 or 2, got p = {p}")

        return grad_o_q_masked

    @staticmethod
    def gradient_o_k_unmasked(q, k, v, o, g, grad, normalize_term, p):
        """Computes gradient of loss with respect to K, in the unmasked case

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            v: (Value tensor) Shape (B, H, N, D)
            o: (Output tensor) Shape (B, H, N, D) output of forward pass
            g: (Denominator tensor) Shape (B, H, N, D) denominator in forward computation
            grad: (Gradient tensor) Shape (B, H, N, D) upstream gradients of Loss w/r/t O
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.

        Returns:
            grad_o_k: Gradient w/r/t K. Shape (B, H, N, D)
        """
        _, _, N, _ = q.shape

        y1 = einops.einsum(1 / g, grad, q, "b h i, b h i j, b h i r -> b h j r")
        z1 = einops.einsum(v, y1, "b h p j, b h j r -> b h p r")

        y2 = einops.einsum(grad, o, "b h i j, b h i j -> b h i")
        z2 = einops.einsum(1 / g, q, y2, "b h i, b h i r, b h i -> b h r")
        z2 = einops.repeat(z2, "b h r -> b h p r", p=N)

        if p == 2:
            y3 = einops.einsum(
                1 / g, grad, q, q, "b h i, b h i j, b h i m, b h i r -> b h j m r"
            )
            z3 = (
                einops.einsum(k, v, y3, "b h p m, b h p j, b h j m r -> b h p r")
                / normalize_term
            )

            y4 = einops.einsum(grad, o, q, "b h i j, b h i j, b h i r -> b h i r")
            z4 = (
                einops.einsum(
                    k, 1 / g, q, y4, "b h p m, b h i, b h i m, b h i r -> b h p r"
                )
                / normalize_term
            )

            grad_o_k = (z1 - z2 + z3 - z4) / normalize_term
        elif p == 1:
            grad_o_k = (z1 - z2) / normalize_term
        else:
            raise ValueError(f"Expected p = 1 or 2. Got p = {p}")

        return grad_o_k

    @staticmethod
    def gradient_o_k_masked(q, k, v, o, g, grad, normalize_term, p):
        """Computes gradient of loss with respect to K, in the masked case

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            v: (Value tensor) Shape (B, H, N, D)
            o: (Output tensor) Shape (B, H, N, D) output of forward pass
            g: (Denominator tensor) Shape (B, H, N, D) denominator in forward computation
            grad: (Gradient tensor) Shape (B, H, N, D) upstream gradients of Loss w/r/t O
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.

        Returns:
            grad_o_k: Gradient w/r/t K. Shape (B, H, N, D)
        """
        _, _, N, _ = q.shape
        # Cursed way to do reverse cumulative sums. Uses advanced indexing since
        # the pytorch "flip" method is very slow. See this PR for more info
        # https://github.com/pytorch/pytorch/issues/16424
        rev_idx = torch.arange(N - 1, -1, -1)
        x1 = einops.einsum(1 / g, grad, q, "b h i, b h i j, b h i r -> b h i j r")
        y1 = torch.cumsum(x1[:, :, rev_idx, :, :], dim=2)[:, :, rev_idx, :, :]
        z1 = einops.einsum(v, y1, "b h p j, b h p j r -> b h p r")

        x2 = einops.einsum(
            1 / g, grad, o, q, "b h i, b h i j, b h i j, b h i r -> b h i j r"
        )
        y2 = torch.cumsum(x2[:, :, rev_idx, :, :], axis=2)[:, :, rev_idx, :, :]
        z2 = einops.reduce(y2, "b h p j r -> b h p r", "sum")

        if p == 2:
            x3 = einops.einsum(
                1 / g, grad, q, q, "b h i, b h i j, b h i m, b h i r -> b h i j m r"
            )
            y3 = torch.cumsum(x3[:, :, rev_idx, :, :, :], 2)[:, :, rev_idx, :, :, :]
            z3 = (
                einops.einsum(k, v, y3, "b h p m, b h p j, b h p j m r -> b h p r")
                / normalize_term
            )

            x4 = einops.einsum(
                1 / g,
                grad,
                q,
                o,
                q,
                "b h i, b h i j, b h i m, b h i j, b h i r -> b h i j m r",
            )
            y4 = torch.cumsum(x4[:, :, rev_idx, :, :, :], 2)[:, :, rev_idx, :, :, :]
            z4 = (
                einops.einsum(k, y4, "b h p m, b h p j m r -> b h p r") / normalize_term
            )
            grad_o_k = (z1 - z2 + z3 - z4) / normalize_term
        elif p == 1:
            grad_o_k = (z1 - z2) / normalize_term
        else:
            raise ValueError(f"Expected p = 1 or 2. Got p = {p}")

        return grad_o_k

    @staticmethod
    def gradient_o_v_unmasked(q, k, g, grad, normalize_term, p):
        """Computes gradient of loss with respect to V, in the unmasked case

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            g: (Denominator tensor) Shape (B, H, N, D) denominator in forward computation
            grad: (Gradient tensor) Shape (B, H, N, D) upstream gradients of Loss w/r/t O
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.

        Returns:
            grad_o_v: Gradient w/r/t V. Shape (B, H, N, D)
        """
        _, _, N, _ = q.shape

        z1 = einops.einsum(1 / g, grad, "b h i, b h i r -> b h r")
        z1 = einops.repeat(z1, "b h r -> b h p r", p=N)

        y2 = einops.einsum(1 / g, grad, q, "b h i, b h i r, b h i m -> b h r m")
        z2 = einops.einsum(k, y2, "b h p m, b h r m -> b h p r") / normalize_term

        if p == 2:
            y3 = einops.einsum(
                1 / g, grad, q, q, "b h i, b h i r, b h i m, b h i l -> b h r m l"
            )
            z3 = einops.einsum(k, k, y3, "b h p m, b h p l, b h r m l -> b h p r") / (
                2 * normalize_term**2
            )
            grad_o_v = z1 + z2 + z3
        elif p == 1:
            grad_o_v = z1 + z2
        else:
            raise ValueError(f"Expected p = 1 or 2, got p = {p}")

        return grad_o_v

    @staticmethod
    def gradient_o_v_masked(q, k, g, grad, normalize_term, p):
        """Computes gradient of loss with respect to V, in the masked case

        For details on the formulas for computation, see Ryan's notes

        Args:
            q: (Query tensor) Shape (B, H, N, D)
            k: (Key tensor) Shape (B, H, N, D)
            g: (Denominator tensor) Shape (B, H, N, D) denominator in forward computation
            grad: (Gradient tensor) Shape (B, H, N, D) upstream gradients of Loss w/r/t O
            normalize_term: (Float) Term to divide score by for normalization
            p: (int) Degree of polynomial approximation to exponential function.
                Restricted to either 1 or 2.

        Returns:
            grad_o_v: Gradient w/r/t V. Shape (B, H, N, D)
        """
        _, _, N, _ = q.shape
        # Cursed way to do reverse cumulative sums. Uses advanced indexing since
        # the pytorch "flip" method is very slow. See this PR for more info
        # https://github.com/pytorch/pytorch/issues/16424
        rev_idx = torch.arange(N - 1, -1, -1)
        y1 = einops.einsum(grad, 1 / g, "b h i r, b h i -> b h i r")
        z1 = torch.cumsum(y1[:, :, rev_idx, :], 2)[:, :, rev_idx, :]

        x2 = einops.einsum(grad, q, 1 / g, "b h i r, b h i m, b h i -> b h i r m")
        y2 = torch.cumsum(x2[:, :, rev_idx, :, :], 2)[:, :, rev_idx, :, :]
        z2 = einops.einsum(k, y2, "b h p m, b h p r m -> b h p r") / normalize_term

        if p == 2:
            x3 = einops.einsum(
                grad, 1 / g, q, q, "b h i r, b h i, b h i m, b h i l -> b h i r m l"
            )
            y3 = torch.cumsum(x3[:, :, rev_idx, :, :, :], 2)[:, :, rev_idx, :, :, :]
            z3 = einops.einsum(k, k, y3, "b h p m, b h p l, b h p r m l -> b h p r") / (
                2 * normalize_term**2
            )
            grad_o_v = z1 + z2 + z3
        elif p == 1:
            grad_o_v = z1 + z2
        else:
            raise ValueError(f"Expected p = 1 or 2, got p = {p}")

        return grad_o_v
