import torch

from tensorpc.apps.mls.backends import tritonstd 
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
flex_attention = torch.compile(flex_attention)

DIM = 128

def _torch_bench_fn_bwd(kwargs):
    q = torch.randn(1, 16, 30720, DIM, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 16, 30720, DIM, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 16, 30720, DIM, dtype=torch.float16, device="cuda")
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    res = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False
    )
    dres = torch.empty_like(res)
    with tritonstd.measure_duration_torch() as dur:

        res.backward(dres)
    return dur.val


def _torch_flex_bench_fn_bwd(kwargs):
    q = torch.randn(1, 16, 30720, DIM, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 16, 30720, DIM, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 16, 30720, DIM, dtype=torch.float16, device="cuda")
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    res = flex_attention(
        q, k, v
    )
    dres = torch.empty_like(res)
    with tritonstd.measure_duration_torch() as dur:

        res.backward(dres)
    return dur.val

def _main():
    for j in range(5):
        dur = _torch_bench_fn_bwd({})
        print(f"torch scaled dot product attention backward: {dur:.3f} ms")
        dur = _torch_flex_bench_fn_bwd({})
        print(f"torch flex attention backward: {dur:.3f} ms")

if __name__ == "__main__":
    _main()