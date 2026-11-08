"""Our own sanity checks (the official data.pt file is not in the ZIP)."""
import torch, math
from diffusion import Diffusion, cosine_schedule
from unet import Unet

torch.manual_seed(0)
T = 50
model = Unet(dim=16, dim_mults=[1, 2, 4, 8])
n_params = sum(p.numel() for p in model.parameters())
print("U-Net parameters:", n_params)
d = Diffusion(model, image_size=32, channels=3, timesteps=T)

# 1) U-Net forward: output must have the same shape as the input
x = torch.randn(4, 3, 32, 32); t = torch.randint(0, T, (4,))
print("1) UNet out shape:", tuple(model(x, t).shape))

# 2) Schedule sanity
ab = d.alphas_cumprod
print("2) alpha_bar first/last:", round(ab[0].item(), 4), ab[-1].item(),
      "| prev[0] =", d.alphas_cumprod_prev[0].item())
assert torch.all(ab[1:] < ab[:-1])

# 3) Posterior coefficients vs. an independent derivation (precision-weighted Gaussians)
#    q(x_{t-1}|x_t,x0) ∝ N(x_t; sqrt(a_t) x_{t-1}, 1-a_t) * N(x_{t-1}; sqrt(abar_{t-1}) x0, 1-abar_{t-1})
for t_ in range(1, T):
    a = d.alphas[t_].double(); abp = d.alphas_cumprod_prev[t_].double()
    prec = a / (1 - a) + 1 / (1 - abp)
    var = 1 / prec
    c1 = (a.sqrt() / (1 - a)) * var
    c2 = (abp.sqrt() / (1 - abp)) * var
    assert abs(var - d.posterior_variance[t_].double()) < 1e-6, t_
    assert abs(c1 - d.posterior_mean_coef1[t_].double()) < 1e-5, t_
    assert abs(c2 - d.posterior_mean_coef2[t_].double()) < 1e-5, t_
print("3) posterior mean/variance match independent derivation: OK")

# 4) q_sample marginal == repeating the one-step rule t times (statistically)
x0 = torch.full((20000, 1, 1, 1), 0.7)
tau = 20
xt = x0.clone()
for i in range(tau + 1):
    xt = d.alphas[i].sqrt() * xt + (1 - d.alphas[i]).sqrt() * torch.randn_like(xt)
direct = d.q_sample(x0, torch.full((20000,), tau), torch.randn_like(x0))
print("4) mean/std iterative:", round(xt.mean().item(), 3), round(xt.std().item(), 3),
      "| direct:", round(direct.mean().item(), 3), round(direct.std().item(), 3))

# 5) Loss runs and is a scalar; backward works
x0 = torch.rand(8, 3, 32, 32) * 2 - 1
loss = d(x0, torch.randn_like(x0)); loss.backward()
print("5) loss (random weights):", round(loss.item(), 4), "| scalar:", loss.dim() == 0)

# 6) Sampling returns images in [0,1]
model.eval()
imgs = d.sample(4)
print("6) sample shape:", tuple(imgs.shape), "range:", round(imgs.min().item(), 3), round(imgs.max().item(), 3))

# 7) Perfect-noise oracle: if the model predicted the TRUE noise, reverse steps must recover x0 well
class Oracle(torch.nn.Module):
    def __init__(s): super().__init__(); s.x0=None; s.sa=None
    def forward(s, xt, t):
        a = d.sqrt_alphas_cumprod[t].view(-1,1,1,1); b = d.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return (xt - a * s.x0) / b
orc = Oracle(); d2 = Diffusion(orc, image_size=8, channels=3, timesteps=T)
orc.x0 = torch.rand(16, 3, 8, 8) * 2 - 1
xt = torch.randn(16, 3, 8, 8)
out = d2.p_sample_loop(xt)   # in [0,1]
err = (out * 2 - 1 - orc.x0).abs().mean().item()
print("7) oracle reverse process mean |x0_rec - x0| =", round(err, 4))
