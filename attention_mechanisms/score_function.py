import torch
import math
import einops


# @torch.jit.script
def fastmax(
    q,
    k,
    v,
    mask=None,
    denum_term=8,
    normalize=0,
    p=1,
    create_attn_matrix=False,
    dropout_rate=0,
):
    """
    Input: query, key, and value matrices (b, h, n, d)
        b: batch size
        h: number of heads
        n: number of tokens
        d: dimension per attention head (d = d_model / h)
    mask: boolean indicating whether to apply causal masking
    denum_term: Hyperparameter to control the standard deviation of <q, k>; stdev(<q, k>) = 1/denum_term
        Stdev of <q, k> is important in general with attention, but even more so when using a taylor
        expansion to approximate an exponential because the error increases with the stdev of <q, k>.
        In normal attention, stdev equates to the "temperature" of the softmax function, and with a
        taylor approximation, higher temperature also means we drift further from the true softmax.
        For positive inputs, this drifting error actually lowers the temperature, and for negative inputs
        it raises the temperature.
    p: can be either 1 or 2 to use the first one or two taylor terms.
    create_attn_matrix: boolean indicating whether to explicitly create the attention matrix. Causes the
        function to become quadratic with N, which we do not want. Should only be used for debugging purposes.
    dropout: dropout rate.
    Output: The result of Attention matrix * Value (b, h, n, d)
    """

    if not create_attn_matrix:
        if normalize == 1:
            denum_term = 1
            q = torch.swapaxes(q, 2, 3)
            k = torch.swapaxes(k, 2, 3)
            q = q - torch.mean(q, dim=3).unsqueeze(-1)
            k = k - torch.mean(k, dim=3).unsqueeze(-1)
            q = torch.swapaxes(q, 2, 3)
            k = torch.swapaxes(k, 2, 3)
            qn = torch.linalg.norm(q, dim=3)
            kn = torch.linalg.norm(k, dim=3)
            q = q / torch.linalg.norm(qn, dim=2, ord=float("inf")).unsqueeze(
                -1
            ).unsqueeze(-1)
            k = k / torch.linalg.norm(kn, dim=2, ord=float("inf")).unsqueeze(
                -1
            ).unsqueeze(-1)
        else:
            denum_term = denum_term * math.sqrt(q.shape[3])
        denum_term2 = 2 * denum_term * denum_term

        # if normalize == 1:
        #     denum_term = 1
        #     q = q - torch.mean(q,dim = 3).unsqueeze(-1)
        #     k = k - torch.mean(k,dim = 3).unsqueeze(-1)
        #     qn = torch.linalg.norm(q, dim = 3)
        #     kn = torch.linalg.norm(k, dim = 3)
        #     q = q/torch.linalg.norm(qn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
        #     k = k/torch.linalg.norm(kn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
        # else:
        #     denum_term = denum_term*math.sqrt(q.shape[3])
        # denum_term2 = 2*denum_term*denum_term

        # Prepare the quadratic terms with respect to k and q:

        if p == 2:
            # Prepare the quadratic terms with respect to k and q:
            k2 = k.unsqueeze(-1) @ k.unsqueeze(
                -2
            )  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
            k2 = k2.flatten(-2)  # (b, h, n, d*d)
            q2 = q.unsqueeze(-1) @ q.unsqueeze(
                -2
            )  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
            q2 = q2.flatten(-2)  # (b, h, n, d*d)
            drop_attn = torch.nn.Dropout(p=dropout_rate)
            k2 = drop_attn(k2)
            q2 = drop_attn(q2)

            if mask is None or not mask:
                first_term = torch.sum(v, -2)  # (b, h, d)

                second_term = (
                    torch.matmul(k.swapaxes(-2, -1), v) / denum_term
                )  # (b, h, d, d)

                third_term = (
                    torch.matmul(k2.swapaxes(-2, -1), v) / denum_term2
                )  # (b, h, d^2, d)

                div1 = (
                    torch.ones([k.shape[0], k.shape[1], 1, 1], device=k.device)
                    * k.shape[2]
                )  # (b, h, 1, 1)
                div2 = torch.sum(k, -2).unsqueeze(-1)  # (b, h, d, 1)
                div3 = torch.sum(k2, -2).unsqueeze(-1)  # (b, h, d^2, 1)

                ans2 = torch.matmul(q, second_term)  # (b, h, n, d)
                ans3 = torch.matmul(q2, third_term)  # (b, h, n, d)
                div2 = torch.matmul(q, div2) / (denum_term)  # (b, h, n, 1)
                div3 = torch.matmul(q2, div3) / (denum_term2)  # (b, h, n, 1)

                ans = ans2 + ans3  # (b, h, n, d)
                ans = torch.add(
                    ans.permute(2, 3, 1, 0), first_term.permute(2, 1, 0)
                ).permute(
                    3, 2, 0, 1
                )  # (b, h, n, d)
                div = div2 + div3  # (b, h, n, d)
                div = torch.add(
                    div.permute(2, 3, 1, 0), div1.permute(3, 2, 1, 0)
                ).permute(
                    3, 2, 0, 1
                )  # (b, h, n, 1)
                ans = ans / div  # (b, h, n, d)

            else:
                # The causal mask implies all terms after i are zeroed out. Thus we can use cumsum to instead of the full summation.
                const_term = torch.cumsum(v, 2)  # (b, h, n, d)
                # Consolidate the interactions between k and v into a single constant matrix to be reused:
                C = torch.einsum(
                    "bhij,bhik -> bhijk", [k, v]
                )  # k,v: (b, h, n, d) -> (b, h, n, d, d)
                # Again, because masked, only take the cumsum of these matrix elements for the linear term:
                C = torch.cumsum(C, 2)  # (b, h, n, d, d)
                # Summation over j for q and C:
                lin_term = (
                    torch.einsum("bhij,bhijk -> bhik", [q, C]) / denum_term
                )  # (b, h, n, d)
                # Consolidate the interactions between k^2 and v into a single constant matrix to be reused:
                D = torch.einsum(
                    "bhij,bhik -> bhijk", [k2, v]
                )  # k2,v: (b, h, n, d) -> (b, h, n, d, d)
                # Again, because masked, only take the cumsum of these matrix elements for the quadratic term:
                D = torch.cumsum(D, 2)  # (b, h, n, d, d)
                quad_term = (
                    torch.einsum("bhij,bhijk -> bhik", [q2, D]) / denum_term2
                )  # (b, h, n, d)

                kcs = torch.cumsum(k, -2)  # (b, h, n, d)
                k2cs = torch.cumsum(k2, -2)  # (b, h, n, d^2)
                div1 = torch.cumsum(
                    torch.ones([q.shape[0], q.shape[1], q.shape[2]], device=k.device), 2
                )  # (b, h, 1)
                div2 = (
                    torch.einsum("bhij,bhij -> bhi", [q, kcs]) / denum_term
                )  # (b, h, n)
                div3 = (
                    torch.einsum("bhij,bhij -> bhi", [q2, k2cs]) / denum_term2
                )  # (b, h, n)
                div = (div1 + div2 + div3).unsqueeze(-1)  # (b, h, n, 1)

                ans = const_term + lin_term + quad_term  # (b, h, n, d)
                ans /= div  # (b, h, n, d)

        # Taylor series with constant and linear terms:
        elif p == 1:
            drop_attn = torch.nn.Dropout(p=dropout_rate)
            k = drop_attn(k)
            q = drop_attn(q)
            if mask is None or not mask:
                first_term = torch.sum(v, -2)  # (b, h, d)
                second_term = (
                    torch.matmul(k.swapaxes(-2, -1), v) / denum_term
                )  # (b, h, d, d)

                div1 = (
                    torch.ones([k.shape[0], k.shape[1], 1, 1], device=k.device)
                    * k.shape[2]
                )  # (b, h, 1, 1)
                div2 = torch.sum(k, -2).unsqueeze(-1)  # (b, h, d, 1)

                ans2 = torch.matmul(q, second_term)  # (b, h, n, d)
                div2 = torch.matmul(q, div2) / (denum_term)  # (b, h, n, 1)

                ans = ans2  # (b, h, n, d)
                ans = torch.add(
                    ans.permute(2, 3, 1, 0), first_term.permute(2, 1, 0)
                ).permute(
                    3, 2, 0, 1
                )  # (b, h, n, d)
                div = div2  # (b, h, n, d)
                div = torch.add(
                    div.permute(2, 3, 1, 0), div1.permute(3, 2, 1, 0)
                ).permute(
                    3, 2, 0, 1
                )  # (b, h, n, 1)
                ans = ans / div  # (b, h, n, d)

            else:
                const_term = torch.cumsum(v, 2)  # (b, h, n, d)
                lin_term = (
                    torch.einsum(
                        "bhij,bhijk -> bhik",
                        [
                            q,
                            torch.cumsum(torch.einsum("bhij,bhik -> bhijk", [k, v]), 2),
                        ],
                    )
                    / denum_term
                )  # (b, h, n, d)

                kcs = torch.cumsum(k, -2)  # (b, h, n, d)
                div1 = torch.cumsum(
                    torch.ones([q.shape[0], q.shape[1], q.shape[2]], device=k.device), 2
                )  # (b, h, 1)
                div2 = (
                    torch.einsum("bhij,bhij -> bhi", [q, kcs]) / denum_term
                )  # (b, h, n)
                div = (div1 + div2).unsqueeze(-1)  # (b, h, n, 1)

                ans = const_term + lin_term  # (b, h, n, d)
                ans /= div  # (b, h, n, d)

        else:
            raise ValueError(f"p must be 1 or 2, got: {p}")

    else:
        denum_term = denum_term * math.sqrt(q.shape[3])
        denum_term2 = 2 * denum_term * denum_term

        k2 = k.unsqueeze(-1) @ k.unsqueeze(
            -2
        )  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
        k2 = k2.flatten(-2)  # (b, h, n, d*d)
        q2 = q.unsqueeze(-1) @ q.unsqueeze(
            -2
        )  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
        q2 = q2.flatten(-2)
        attn = (
            1
            + torch.matmul(q, torch.swapaxes(k, -2, -1)) / denum_term
            + torch.matmul(q2, torch.swapaxes(k2, -2, -1)) / denum_term2
        )
        if mask is not None:
            attn = torch.where(mask == 0, 0, attn)
        attn /= (torch.sum(attn, axis=3)).unsqueeze(-1)
        ans = torch.matmul(attn, v)
        return ans, attn
    return ans
