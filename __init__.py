"""Give comfy_kitchen's rms_rope_split_half a torch autograd path.

The CUDA kernel has no backward, so TrainLoraNode dies on Anima/Qwen with
"no autograd formula was registered". Outside training the kernel is kept.
"""
import torch


def _fallback(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    def one(x, scale):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
        if scale is not None:
            x = x * scale
        return x

    q = one(q, q_scale)
    k = one(k, k_scale) if k_scale is not None else one(k, None)

    def rope(x):
        d = x.shape[-1]
        r = rot_dim if rot_dim else d
        half = r // 2
        x1, x2 = x[..., :half], x[..., half:r]
        f = freqs_cis
        fr, fi = (f.real, f.imag) if torch.is_complex(f) else (f[..., 0], f[..., 1])
        fr, fi = fr.to(x.dtype), fi.to(x.dtype)
        while fr.ndim < x1.ndim:
            fr, fi = fr.unsqueeze(-2), fi.unsqueeze(-2)
        while fr.shape[-2] != x1.shape[-2]:
            fr, fi = fr.unsqueeze(-3), fi.unsqueeze(-3)
        return torch.cat([x1 * fr - x2 * fi, x1 * fi + x2 * fr, x[..., r:]], dim=-1)

    return rope(q), rope(k)


def patch():
    try:
        import comfy.quant_ops as qo
        ck = qo.ck
    except Exception:
        return
    if getattr(ck, "_vnccs_grad_patch", False):
        return
    orig = ck.rms_rope_split_half

    def wrapper(*args, **kwargs):
        if torch.is_grad_enabled():
            return _fallback(*args, **kwargs)
        return orig(*args, **kwargs)

    ck.rms_rope_split_half = wrapper
    ck._vnccs_grad_patch = True


patch()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
