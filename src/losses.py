import torch
import torch.nn.functional as F


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x_hat, x)


def charbonnier_loss(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = x_hat - x
    return torch.sqrt(diff * diff + eps * eps).mean()


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return kx, ky


def sobel_edges(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx, ky = _sobel_kernels(x.device, x.dtype)
    return F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)


def sobel_edge_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    hx, hy = sobel_edges(x_hat)
    tx, ty = sobel_edges(x)
    return F.l1_loss(hx, tx) + F.l1_loss(hy, ty)


def gradient_difference_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    pred_dx = x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]
    pred_dy = x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]
    target_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    target_dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def simple_multiscale_l1(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    loss = F.l1_loss(x_hat, x)
    for scale, weight in [(2, 0.5), (4, 0.25)]:
        pred = F.avg_pool2d(x_hat, kernel_size=scale, stride=scale)
        target = F.avg_pool2d(x, kernel_size=scale, stride=scale)
        loss = loss + weight * F.l1_loss(pred, target)
    return loss


def differentiable_ssim_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    window_size: int = 7,
    data_range: float = 1.0,
) -> torch.Tensor:
    pad = int(window_size) // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(x_hat, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(x_hat * x_hat, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * x_hat, window_size, stride=1, padding=pad) - mu_x * mu_y
    numer = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denom = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = numer / denom.clamp_min(1e-12)
    return 1.0 - ssim.clamp(0.0, 1.0).mean()


def multiscale_ssim_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    loss = differentiable_ssim_loss(x_hat, x)
    for scale, weight in [(2, 0.5), (4, 0.25)]:
        pred = F.avg_pool2d(x_hat, kernel_size=scale, stride=scale)
        target = F.avg_pool2d(x, kernel_size=scale, stride=scale)
        loss = loss + weight * differentiable_ssim_loss(pred, target)
    return loss


def frequency_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    pred = torch.fft.rfft2(x_hat.float(), norm="ortho")
    target = torch.fft.rfft2(x.float(), norm="ortho")
    pred_mag = torch.log1p(torch.abs(pred))
    target_mag = torch.log1p(torch.abs(target))
    return F.l1_loss(pred_mag, target_mag)


def data_consistency_loss(
    measurement, x_hat: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    pred_y = measurement.A_forward(measurement.flatten_img(x_hat))
    return F.mse_loss(pred_y, y)


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w


def discriminator_wgan_loss(
    real_scores: torch.Tensor, fake_scores: torch.Tensor
) -> torch.Tensor:
    return fake_scores.mean() - real_scores.mean()


def generator_adversarial_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()


def gradient_penalty(
    discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = real.device
    batch_size = real.shape[0]
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    interp = eps * real + (1.0 - eps) * fake
    interp.requires_grad_(True)

    scores = discriminator(interp)
    grad_outputs = torch.ones_like(scores)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.reshape(batch_size, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
