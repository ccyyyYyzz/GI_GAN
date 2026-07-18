from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from measurement_conditioned_vqgan import VQAutoencoder
from src.dc_balanced import hadamard_lowsequency_non_dc_rows


DEFAULT_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/completion_gan_round18/pqbf_pilot_seed0/cache"
)
DEFAULT_VQGAN = Path(
    "E:/ns_mc_gan_gi_code_fcc_phase1/outputs/compatibility/"
    "measurement_conditioned_vqgan/prior_multiseed_hashclean_seed0/"
    "vqgan_continuation/checkpoints/vqgan_continuation_best_by_lpips.pt"
)
DEFAULT_VQAE = Path(
    "E:/ns_mc_gan_gi_code_fcc_phase1/outputs/compatibility/"
    "measurement_conditioned_vqgan/prior_multiseed_hashclean_seed0/"
    "vqae_continuation/checkpoints/vqae_continuation_best_by_lpips.pt"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round29/"
    "vqgan_secant_binary_geometry"
)


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def load_prior(path: Path, device: torch.device) -> tuple[VQAutoencoder, dict]:
    payload = torch.load(path, map_location=device)
    config = payload["config"]
    model_cfg = config["model"]
    model = VQAutoencoder(
        codebook_size=int(model_cfg["codebook_size"]),
        z_dim=int(model_cfg["z_dim"]),
        base=int(model_cfg["base_channels"]),
        beta=float(model_cfg.get("commit_beta", 0.25)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


@torch.no_grad()
def reconstruct(model: VQAutoencoder, images: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, images.shape[0], int(batch_size)):
        stop = min(images.shape[0], start + int(batch_size))
        prediction, _indices, _vq_loss, _stats = model(images[start:stop])
        output.append(prediction.float())
    return torch.cat(output)


def balanced_ste(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    n = logits.shape[1]
    if n % 2:
        raise ValueError("BALANCED_ROWS_REQUIRE_EVEN_DIMENSION")
    centered = logits - logits.mean(dim=1, keepdim=True)
    soft = torch.tanh(centered / max(float(temperature), 1.0e-4))
    top = torch.topk(centered, k=n // 2, dim=1, largest=True, sorted=False).indices
    hard = -torch.ones_like(centered)
    hard.scatter_(1, top, 1.0)
    return (hard + soft - soft.detach()) / math.sqrt(n)


def sample_normalized_secants(
    images: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    total = images.shape[0]
    first = torch.randint(total, (int(count),), generator=generator, device=images.device)
    offset = torch.randint(1, total, (int(count),), generator=generator, device=images.device)
    second = (first + offset) % total
    difference = images[first].flatten(1) - images[second].flatten(1)
    return difference / difference.norm(dim=1, keepdim=True).clamp_min(1.0e-8)


def off_diagonal_gram_loss(rows: torch.Tensor) -> torch.Tensor:
    gram = rows @ rows.t()
    identity = torch.eye(rows.shape[0], device=rows.device, dtype=rows.dtype)
    return ((gram - identity) * (1.0 - identity)).square().mean()


def optimize_code(
    images: torch.Tensor,
    *,
    m: int,
    steps: int,
    pair_batch: int,
    lr: float,
    temperature_start: float,
    temperature_end: float,
    softmin_tau: float,
    coherence_weight: float,
    mean_weight: float,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    device = images.device
    n = int(images.shape[-2] * images.shape[-1])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    logits = torch.nn.Parameter(torch.randn((int(m), n), generator=generator, device=device))
    optimizer = torch.optim.Adam([logits], lr=float(lr), betas=(0.5, 0.9))
    log: list[dict[str, float]] = []
    for step in range(int(steps)):
        progress = step / max(1, int(steps) - 1)
        temperature = float(temperature_start) * (
            float(temperature_end) / float(temperature_start)
        ) ** progress
        secants = sample_normalized_secants(
            images, count=int(pair_batch), generator=generator
        )
        rows = balanced_ste(logits, temperature)
        energy = (secants @ rows.t()).square().sum(dim=1)
        tau = float(softmin_tau)
        soft_min = -tau * (
            torch.logsumexp(-energy / tau, dim=0) - math.log(energy.numel())
        )
        coherence = off_diagonal_gram_loss(rows)
        loss = (
            -soft_min
            - float(mean_weight) * energy.mean()
            + float(coherence_weight) * coherence
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], 10.0)
        optimizer.step()
        if step % 20 == 0 or step + 1 == int(steps):
            log.append(
                {
                    "step": float(step + 1),
                    "temperature": temperature,
                    "loss": float(loss.detach().cpu()),
                    "soft_min": float(soft_min.detach().cpu()),
                    "energy_mean": float(energy.mean().detach().cpu()),
                    "energy_q05": float(torch.quantile(energy, 0.05).detach().cpu()),
                    "coherence_loss": float(coherence.detach().cpu()),
                }
            )
    with torch.no_grad():
        rows = balanced_ste(logits, float(temperature_end)).detach()
    return rows, log


@torch.no_grad()
def evaluate_code(
    rows: torch.Tensor,
    images: torch.Tensor,
    *,
    pairs: int,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator(device=images.device)
    generator.manual_seed(int(seed))
    secants = sample_normalized_secants(images, count=int(pairs), generator=generator)
    energy = (secants @ rows.t()).square().sum(dim=1)
    gram = rows @ rows.t()
    mask = ~torch.eye(rows.shape[0], device=rows.device, dtype=torch.bool)
    off = gram[mask].abs()
    return {
        "pairs": float(pairs),
        "energy_min": float(energy.min().cpu()),
        "energy_q01": float(torch.quantile(energy, 0.01).cpu()),
        "energy_q05": float(torch.quantile(energy, 0.05).cpu()),
        "energy_median": float(energy.median().cpu()),
        "energy_mean": float(energy.mean().cpu()),
        "energy_q95": float(torch.quantile(energy, 0.95).cpu()),
        "row_coherence_mean_abs": float(off.mean().cpu()),
        "row_coherence_max_abs": float(off.max().cpu()),
        "row_balance_max_abs": float(rows.sum(dim=1).abs().max().cpu()),
        "row_norm_max_error": float((rows.norm(dim=1) - 1.0).abs().max().cpu()),
        "unique_rows": float(torch.unique(rows, dim=0).shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--vqgan", type=Path, default=DEFAULT_VQGAN)
    parser.add_argument("--vqae", type=Path, default=DEFAULT_VQAE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--val-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--m", type=int, default=205)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pair-batch", type=int, default=128)
    parser.add_argument("--eval-pairs", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.15)
    parser.add_argument("--softmin-tau", type=float, default=0.01)
    parser.add_argument("--coherence-weight", type=float, default=0.5)
    parser.add_argument("--mean-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    set_seed(args.seed)
    started = time.time()
    device = torch.device("cuda")
    train_payload = torch.load(args.cache_dir / "train.pt", map_location="cpu")
    val_payload = torch.load(args.cache_dir / "val.pt", map_location="cpu")
    train_truth = train_payload["tensors"]["truth"][: int(args.train_count)].float().to(device)
    val_truth = val_payload["tensors"]["truth"][: int(args.val_count)].float().to(device)
    vqgan, gan_payload = load_prior(args.vqgan, device)
    vqae, ae_payload = load_prior(args.vqae, device)
    train_sets = {
        "vqgan": reconstruct(vqgan, train_truth, batch_size=int(args.batch_size)),
        "vqae": reconstruct(vqae, train_truth, batch_size=int(args.batch_size)),
    }
    validation_sets = {
        "real": val_truth,
        "vqgan": reconstruct(vqgan, val_truth, batch_size=int(args.batch_size)),
        "vqae": reconstruct(vqae, val_truth, batch_size=int(args.batch_size)),
    }

    n = int(val_truth.shape[-2] * val_truth.shape[-1])
    rng = np.random.default_rng(int(args.seed) + 77)
    random_rows = np.empty((int(args.m), n), dtype=np.float32)
    template = np.concatenate(
        [-np.ones(n // 2, dtype=np.float32), np.ones(n // 2, dtype=np.float32)]
    )
    for index in range(int(args.m)):
        random_rows[index] = rng.permutation(template)
    random_rows = torch.from_numpy(random_rows / math.sqrt(n)).to(device)
    hadamard_rows = torch.from_numpy(
        hadamard_lowsequency_non_dc_rows(int(args.m), n)
    ).to(device=device, dtype=torch.float32)

    codes = {"random_balanced": random_rows, "hadamard_lowsequency": hadamard_rows}
    optimization_logs: dict[str, list[dict[str, float]]] = {}
    for source_index, (source, images) in enumerate(train_sets.items()):
        rows, log = optimize_code(
            images,
            m=int(args.m),
            steps=int(args.steps),
            pair_batch=int(args.pair_batch),
            lr=float(args.lr),
            temperature_start=float(args.temperature_start),
            temperature_end=float(args.temperature_end),
            softmin_tau=float(args.softmin_tau),
            coherence_weight=float(args.coherence_weight),
            mean_weight=float(args.mean_weight),
            seed=int(args.seed) + 1000 * source_index,
        )
        name = f"learned_on_{source}_secants"
        codes[name] = rows
        optimization_logs[name] = log

    evaluation: dict[str, dict[str, dict[str, float]]] = {}
    for code_name, rows in codes.items():
        evaluation[code_name] = {}
        for target_index, (target_name, images) in enumerate(validation_sets.items()):
            evaluation[code_name][target_name] = evaluate_code(
                rows,
                images,
                pairs=int(args.eval_pairs),
                seed=int(args.seed) + 9000 + 100 * target_index,
            )

    payload = {
        "status": "VQGAN_SECANT_BALANCED_BINARY_GEOMETRY_PILOT",
        "validation_only": True,
        "test_split_opened": False,
        "m": int(args.m),
        "n": n,
        "sampling_ratio": float(int(args.m) / n),
        "train_count": int(train_truth.shape[0]),
        "val_count": int(val_truth.shape[0]),
        "vqgan_step": int(gan_payload["step"]),
        "vqae_step": int(ae_payload["step"]),
        "evaluation": evaluation,
        "optimization_logs": optimization_logs,
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"codes": {name: rows.cpu() for name, rows in codes.items()}}, args.output_dir / "codes.pt")
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
