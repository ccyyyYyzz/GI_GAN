import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResidualUNetGenerator(nn.Module):
    """Small residual U-Net that predicts only the null-space residual."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(2, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.bottleneck = ConvBlock(128, 256)

        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(
        self, x_data: torch.Tensor, noise_map: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise_map is not None:
            x = torch.cat([x_data, noise_map], dim=1)
        else:
            x = x_data
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("Generator expects [B, 2, H, W] or x_data plus noise_map.")

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out(d1)


class PlainUNetGenerator(nn.Module):
    """Plain U-Net residual proposer for architecture ablation."""

    model_type = "unet"

    def __init__(self, base_channels: int = 48) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1 = ConvBlock(2, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x_data: torch.Tensor, noise_map: torch.Tensor | None = None, y=None) -> torch.Tensor:
        del y
        x = torch.cat([x_data, noise_map], dim=1) if noise_map is not None else x_data
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("PlainUNetGenerator expects x_data plus noise_map.")
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out(d1)


class PatchDiscriminator(nn.Module):
    """Patch critic for WGAN-GP. The output is raw logits, not probabilities."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.proj(x))


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth: int | None = None) -> None:
        super().__init__()
        growth = int(growth or max(16, channels // 2))
        groups = min(8, channels)
        self.conv1 = nn.Conv2d(channels, growth, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth, growth, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.act(self.conv1(x))
        y2 = self.act(self.conv2(torch.cat([x, y1], dim=1)))
        y3 = self.norm(self.conv3(torch.cat([x, y1, y2], dim=1)))
        return self.act(x + 0.2 * y3)


class WideUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ResidualConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResidualUNetGeneratorWide(nn.Module):
    """Wider residual U-Net that still predicts a null-space residual."""

    model_type = "residual_unet_wide"

    def __init__(self, base_channels: int = 64, attention_gate: bool = False) -> None:
        super().__init__()
        del attention_gate
        c = int(base_channels)
        self.enc1 = ResidualConvBlock(2, c)
        self.enc2 = ResidualConvBlock(c, c * 2)
        self.enc3 = ResidualConvBlock(c * 2, c * 4)
        self.enc4 = ResidualConvBlock(c * 4, c * 8)
        self.bottleneck = ResidualConvBlock(c * 8, c * 8)
        self.up4 = WideUpBlock(c * 8, c * 8, c * 4)
        self.up3 = WideUpBlock(c * 4, c * 4, c * 2)
        self.up2 = WideUpBlock(c * 2, c * 2, c)
        self.up1 = WideUpBlock(c, c, c)
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x_data: torch.Tensor, noise_map: torch.Tensor | None = None, y=None):
        del y
        x = torch.cat([x_data, noise_map], dim=1) if noise_map is not None else x_data
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("ResidualUNetGeneratorWide expects x_data plus noise_map.")
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e4 = self.enc4(F.avg_pool2d(e3, 2))
        b = self.bottleneck(F.avg_pool2d(e4, 2))
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out(d1)


class ResUNetGenerator(ResidualUNetGeneratorWide):
    """Residual U-Net alias with an explicit Phase 25 architecture label."""

    model_type = "resunet"


class NAFBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = int(channels) * int(expansion)
        self.norm1 = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.ffn2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(self.pw1(self.norm1(x)))
        y1, y2 = y.chunk(2, dim=1)
        y = self.pw2(y1 * y2)
        x = x + self.beta * y
        y = self.ffn1(self.norm2(x))
        y1, y2 = y.chunk(2, dim=1)
        y = self.ffn2(y1 * y2)
        return x + self.gamma * y


class NAFNetSmallGenerator(nn.Module):
    """Small NAFNet-style residual proposer used only as G_theta."""

    model_type = "nafnet_small"

    def __init__(self, base_channels: int = 48, num_blocks: int = 8) -> None:
        super().__init__()
        c = int(base_channels)
        self.head = nn.Conv2d(2, c, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(c) for _ in range(int(num_blocks))])
        self.tail = nn.Conv2d(c, 1, kernel_size=3, padding=1)

    def forward(self, x_data: torch.Tensor, noise_map: torch.Tensor | None = None, y=None) -> torch.Tensor:
        del y
        x = torch.cat([x_data, noise_map], dim=1) if noise_map is not None else x_data
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("NAFNetSmallGenerator expects x_data plus noise_map.")
        feat = self.head(x)
        return self.tail(self.body(feat))


class UnrolledISTAGenerator(nn.Module):
    """Learned unrolled residual proposer, followed by the shared Pi_y outside."""

    model_type = "unrolled_ista"

    def __init__(self, base_channels: int = 48, steps: int = 5) -> None:
        super().__init__()
        c = int(base_channels)
        self.steps = int(steps)
        self.enc = nn.Conv2d(2, c, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualConvBlock(c, c) for _ in range(self.steps)])
        self.updates = nn.ModuleList([nn.Conv2d(c, 1, kernel_size=3, padding=1) for _ in range(self.steps)])
        self.step_scale = nn.Parameter(torch.full((self.steps,), 0.1))

    def forward(self, x_data: torch.Tensor, noise_map: torch.Tensor | None = None, y=None) -> torch.Tensor:
        del y
        x = torch.cat([x_data, noise_map], dim=1) if noise_map is not None else x_data
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("UnrolledISTAGenerator expects x_data plus noise_map.")
        feat = self.enc(x)
        residual = torch.zeros_like(x_data)
        for idx, (block, update) in enumerate(zip(self.blocks, self.updates)):
            feat = block(feat)
            residual = residual + self.step_scale[idx] * update(feat)
        return residual


class ResidualRefinerNet(nn.Module):
    model_type = "residual_refiner"

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            ResidualConvBlock(3, c),
            ResidualConvBlock(c, c),
            ResidualConvBlock(c, c),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward(self, x_data: torch.Tensor, x_hat_stage1: torch.Tensor) -> torch.Tensor:
        err = torch.abs(x_hat_stage1 - x_data)
        return self.net(torch.cat([x_data, x_hat_stage1, err], dim=1))


class DirectYToImageBaseline(nn.Module):
    """Direct measurement-to-image baseline, followed by dc_project outside."""

    model_type = "direct_y_to_image"
    output_kind = "direct_image"

    def __init__(self, m: int, base_channels: int = 128, img_size: int = 64) -> None:
        super().__init__()
        if img_size != 64:
            raise ValueError("DirectYToImageBaseline currently expects img_size=64.")
        c = int(base_channels)
        self.fc = nn.Sequential(
            nn.Linear(int(m), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 8 * 8 * c),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            ResidualConvBlock(c, c),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualConvBlock(c, c // 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualConvBlock(c // 2, c // 4),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualConvBlock(c // 4, max(16, c // 8)),
            nn.Conv2d(max(16, c // 8), 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.img_size = int(img_size)
        self.base_channels = c

    def forward(self, x_data=None, noise_map=None, y: torch.Tensor | None = None):
        del x_data, noise_map
        if y is None:
            raise ValueError("DirectYToImageBaseline requires measurement coefficients y.")
        feat = self.fc(y).view(y.shape[0], self.base_channels, 8, 8)
        return self.decoder(feat)


class BackprojectionEnhancer(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            ResidualConvBlock(1, c),
            ResidualConvBlock(c, c),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward(self, x_data: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_data + self.net(x_data), 0.0, 1.0)


class HQResidualUNet(nn.Module):
    """High-capacity residual U-Net tuned for PSNR/SSIM reconstruction."""

    model_type = "hq_unet"

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc0 = nn.Sequential(ResidualConvBlock(2, c), ResidualDenseBlock(c))
        self.enc1 = nn.Sequential(ResidualConvBlock(c, c * 2), ResidualDenseBlock(c * 2))
        self.enc2 = nn.Sequential(ResidualConvBlock(c * 2, c * 4), ResidualDenseBlock(c * 4))
        self.enc3 = nn.Sequential(ResidualConvBlock(c * 4, c * 8), ResidualDenseBlock(c * 8))
        self.bottleneck = nn.Sequential(ResidualConvBlock(c * 8, c * 8), ResidualDenseBlock(c * 8))
        self.up3 = WideUpBlock(c * 8, c * 8, c * 4)
        self.dec3 = ResidualDenseBlock(c * 4)
        self.up2 = WideUpBlock(c * 4, c * 4, c * 2)
        self.dec2 = ResidualDenseBlock(c * 2)
        self.up1 = WideUpBlock(c * 2, c * 2, c)
        self.dec1 = ResidualDenseBlock(c)
        self.up0 = WideUpBlock(c, c, c)
        self.dec0 = ResidualDenseBlock(c)
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x_data: torch.Tensor, noise_map: torch.Tensor | None = None, y=None):
        del y
        x = torch.cat([x_data, noise_map], dim=1) if noise_map is not None else x_data
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("HQResidualUNet expects x_data plus noise_map.")
        e0 = self.enc0(x)
        e1 = self.enc1(F.avg_pool2d(e0, 2))
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))
        d3 = self.dec3(self.up3(b, e3))
        d2 = self.dec2(self.up2(d3, e2))
        d1 = self.dec1(self.up1(d2, e1))
        d0 = self.dec0(self.up0(d1, e0))
        return self.out(d0)


class HQRefiner(nn.Module):
    model_type = "hq_refiner"

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            ResidualConvBlock(3, c),
            ResidualDenseBlock(c),
            ResidualDenseBlock(c),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward(self, x_data: torch.Tensor, x_hat_stage1: torch.Tensor) -> torch.Tensor:
        proxy = torch.abs(x_hat_stage1 - x_data)
        return self.net(torch.cat([x_data, x_hat_stage1, proxy], dim=1))


class HQTwoStageReconstructor(nn.Module):
    model_type = "hq_two_stage"

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.stage1 = HQResidualUNet(base_channels=base_channels)
        self.refiner = HQRefiner(base_channels=base_channels)

    def forward(self, x_data: torch.Tensor, noise_map: torch.Tensor | None = None, y=None):
        return self.stage1(x_data, noise_map, y=y)

    def refine(self, x_data: torch.Tensor, x_hat_stage1: torch.Tensor) -> torch.Tensor:
        return self.refiner(x_data, x_hat_stage1)


class DirectCoeffToImageNet(DirectYToImageBaseline):
    model_type = "direct_coeff_to_image"


def build_generator(config: dict, measurement=None) -> nn.Module:
    model_type = str(config.get("model_type", "residual_unet_small")).lower()
    base_channels = int(config.get("base_channels", 64))
    if model_type == "unet":
        return PlainUNetGenerator(base_channels=base_channels)
    if model_type == "resunet":
        return ResUNetGenerator(
            base_channels=base_channels,
            attention_gate=bool(config.get("attention_gate", False)),
        )
    if model_type == "nafnet_small":
        return NAFNetSmallGenerator(
            base_channels=int(config.get("nafnet_channels", base_channels)),
            num_blocks=int(config.get("nafnet_blocks", 8)),
        )
    if model_type == "unrolled_ista":
        return UnrolledISTAGenerator(
            base_channels=base_channels,
            steps=int(config.get("unrolled_ista_steps", 5)),
        )
    if model_type == "residual_unet_small":
        model = ResidualUNetGenerator()
        model.model_type = model_type
        return model
    if model_type == "residual_unet_wide":
        return ResidualUNetGeneratorWide(
            base_channels=base_channels,
            attention_gate=bool(config.get("attention_gate", False)),
        )
    if model_type == "residual_unet_wide_refiner":
        # The Phase 8 refiner path is represented as a two-stage module using
        # the wide U-Net stage and lightweight refiner.
        module = HQTwoStageReconstructor(base_channels=base_channels)
        module.stage1 = ResidualUNetGeneratorWide(base_channels=base_channels)
        module.refiner = ResidualRefinerNet(base_channels=base_channels)
        module.model_type = model_type
        return module
    if model_type == "hq_unet":
        return HQResidualUNet(base_channels=base_channels)
    if model_type == "hq_two_stage":
        return HQTwoStageReconstructor(base_channels=base_channels)
    if model_type in {"direct_y_to_image", "direct_coeff_to_image"}:
        if measurement is None:
            raise ValueError("Direct measurement-to-image models require measurement.m.")
        cls = DirectCoeffToImageNet if model_type == "direct_coeff_to_image" else DirectYToImageBaseline
        return cls(
            m=getattr(measurement, "m"),
            base_channels=int(config.get("direct_base_channels", 128)),
            img_size=int(config.get("img_size", 64)),
        )
    raise ValueError(
        "model_type must be one of: unet, resunet, nafnet_small, unrolled_ista, "
        "residual_unet_small, residual_unet_wide, "
        "residual_unet_wide_refiner, hq_unet, hq_two_stage, direct_y_to_image, "
        "direct_coeff_to_image."
    )
