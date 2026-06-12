import numpy as np
import torch

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except Exception:
    peak_signal_noise_ratio = None
    structural_similarity = None


def mse(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    return torch.mean((x_hat - x) ** 2).item()


def psnr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    if peak_signal_noise_ratio is None:
        return _fallback_psnr_torch(x_hat, x)
    try:
        return _mean_skimage_metric(x_hat, x, peak_signal_noise_ratio)
    except Exception:
        return _fallback_psnr_torch(x_hat, x)


def ssim(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    if structural_similarity is None:
        return _fallback_ssim_torch(x_hat, x)
    try:
        return _mean_skimage_metric(x_hat, x, structural_similarity)
    except Exception:
        return _fallback_ssim_torch(x_hat, x)


def relative_measurement_error(measurement, x_hat: torch.Tensor, y: torch.Tensor) -> float:
    pred_y = measurement.A_forward(measurement.flatten_img(x_hat))
    numer = torch.linalg.norm(pred_y - y, dim=1)
    denom = torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    return (numer / denom).mean().item()


def batch_metrics(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    measurement=None,
    y: torch.Tensor | None = None,
) -> dict[str, float]:
    result = {
        "mse": mse(x_hat, x),
        "psnr": psnr(x_hat, x),
        "ssim": ssim(x_hat, x),
    }
    if measurement is not None and y is not None:
        result["rel_meas_error"] = relative_measurement_error(measurement, x_hat, y)
    return result


def _mean_skimage_metric(x_hat: torch.Tensor, x: torch.Tensor, fn) -> float:
    x_hat_np = _to_numpy_images(x_hat)
    x_np = _to_numpy_images(x)
    values = []
    for pred, target in zip(x_hat_np, x_np):
        if fn is peak_signal_noise_ratio:
            values.append(fn(target, pred, data_range=1.0))
        else:
            values.append(fn(target, pred, data_range=1.0))
    return float(np.mean(values))


def _mean_numpy_metric(x_hat: torch.Tensor, x: torch.Tensor, fn) -> float:
    x_hat_np = _to_numpy_images(x_hat)
    x_np = _to_numpy_images(x)
    return float(np.mean([fn(target, pred) for pred, target in zip(x_hat_np, x_np)]))


def _to_numpy_images(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(0.0, 1.0).cpu().float().numpy()
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    return x


def _fallback_psnr(target: np.ndarray, pred: np.ndarray) -> float:
    err = float(np.mean((target - pred) ** 2))
    if err <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / err))


def _fallback_ssim(target: np.ndarray, pred: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = float(target.mean())
    mu_y = float(pred.mean())
    sigma_x = float(target.var())
    sigma_y = float(pred.var())
    sigma_xy = float(((target - mu_x) * (pred - mu_y)).mean())
    numer = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denom = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return numer / max(denom, 1e-12)


def _fallback_psnr_torch(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    x_hat = x_hat.detach().clamp(0.0, 1.0).float()
    x = x.detach().clamp(0.0, 1.0).float()
    err = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3))
    psnr_values = torch.where(
        err <= 1e-12,
        torch.full_like(err, float("inf")),
        10.0 * torch.log10(1.0 / err.clamp_min(1e-12)),
    )
    return psnr_values.mean().item()


def _fallback_ssim_torch(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    x_hat = x_hat.detach().clamp(0.0, 1.0).float()
    x = x.detach().clamp(0.0, 1.0).float()
    dims = (1, 2, 3)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = x.mean(dim=dims)
    mu_y = x_hat.mean(dim=dims)
    sigma_x = x.var(dim=dims, unbiased=False)
    sigma_y = x_hat.var(dim=dims, unbiased=False)
    sigma_xy = ((x - mu_x[:, None, None, None]) * (x_hat - mu_y[:, None, None, None])).mean(
        dim=dims
    )
    numer = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denom = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return (numer / denom.clamp_min(1e-12)).mean().item()
