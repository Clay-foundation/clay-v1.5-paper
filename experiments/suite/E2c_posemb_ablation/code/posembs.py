"""Position codes for the E2c arms. All return [h*w, dim] (same interface as claymodel.utils)."""
import torch
from claymodel.utils import posemb_sincos_2d

D_EFF = 1016  # Clay v1.5 ViT-L calls the code with dim = 1024 - 8 = 1016

def pe_A_implemented(h, w, dim, gsd=1.0, temperature=10000, dtype=torch.float32):
    """Clay's posemb_sincos_2d_with_gsd, verbatim except the exponent denominator is ViT-L's 1016
    (so the frequency spread is the released model's: all within 1.8 % of gsd rad/token)."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    gsd = torch.as_tensor(gsd).float().cpu()
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / D_EFF)) * (gsd / 1.0)
    y = y.flatten()[:, None] * omega[None, :]; x = x.flatten()[:, None] * omega[None, :]
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1).type(dtype)

def pe_B_standard(h, w, dim, gsd=1.0, temperature=10000, dtype=torch.float32):
    """Standard 2-D sin-cos (MAE / simple-ViT), no GSD."""
    return posemb_sincos_2d(h, w, dim, temperature, dtype)

GSD_REF = 10.0
def pe_C_scaled(h, w, dim, gsd=1.0, temperature=10000, dtype=torch.float32):
    """Scale-MAE-style: standard frequencies, positions multiplied by gsd / 10 m."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    s = float(torch.as_tensor(gsd)) / GSD_REF
    omega = torch.arange(dim // 4) / (dim // 4 - 1); omega = 1.0 / (temperature ** omega)
    y = (y.flatten()[:, None] * s) * omega[None, :]; x = (x.flatten()[:, None] * s) * omega[None, :]
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1).type(dtype)

ARMS = {"A": pe_A_implemented, "B": pe_B_standard, "C": pe_C_scaled}
