# VQGAN Multi-Seed Pareto Confirmation

Classification: `VQGAN_PRIOR_TRANSFER_CONFIRMED_MULTI_SEED`
Development gate passed: `True`

## Quality Mode
- lpips: mean delta VQGAN-VQAE = -0.126208, 95% cluster CI [-0.130222, -0.122006], per-seed {'0': -0.12851227746432414, '1': -0.12415958011115436, '2': -0.12595267307915492}
- rapsd: mean delta VQGAN-VQAE = -0.00113292, 95% cluster CI [-0.00138029, -0.000970869], per-seed {'0': -0.0010048209804921133, '1': -0.0010084612926412975, '2': -0.0013854689181275206}
- psnr: mean delta VQGAN-VQAE = -1.70134, 95% cluster CI [-1.91504, -1.56568], per-seed {'0': -1.5833039319814701, '1': -1.596739275678041, '2': -1.923981683642038}
- full_rmse: mean delta VQGAN-VQAE = 0.0163863, 95% cluster CI [0.0148343, 0.0187815], per-seed {'0': 0.014954189464333469, '1': 0.01532006709385314, '2': 0.018884544952015858}
- centered_rmse: mean delta VQGAN-VQAE = 0.016384, 95% cluster CI [0.014831, 0.0188384], per-seed {'0': 0.014952061454096112, '1': 0.015317617715481896, '2': 0.018882387063058566}
- relmeaserr: mean delta VQGAN-VQAE = 9.10815e-10, 95% cluster CI [-9.29744e-10, 2.67772e-09], per-seed {'0': -9.868103556520659e-11, '1': 8.573908250042448e-10, '2': 1.9737366463434833e-09}
- method means: {
  "vqae": {
    "lpips": 0.29315104874937487,
    "rapsd": 0.003090364859557352,
    "psnr": 22.868842917091083,
    "ssim": 0.6580807993313126,
    "full_rmse": 0.07492476351520354,
    "centered_rmse": 0.07492137539520623,
    "relmeaserr": 2.2009982770613648e-07
  },
  "vqgan": {
    "lpips": 0.16694287186449705,
    "rapsd": 0.0019574477958037074,
    "psnr": 21.167501286657235,
    "ssim": 0.5727813237731159,
    "full_rmse": 0.09131103068527102,
    "centered_rmse": 0.09130539747275175,
    "relmeaserr": 2.2101064318473065e-07
  }
}

## Balanced Mode
- lpips: mean delta VQGAN-VQAE = -0.0686017, 95% cluster CI [-0.0750664, -0.0622031], per-seed {'0': -0.06838001501455437, '1': -0.06248262617737055, '2': -0.07494252515607513}
- rapsd: mean delta VQGAN-VQAE = 0.000365449, 95% cluster CI [0.000282429, 0.000436428], per-seed {'0': 0.00039058711542519476, '1': 0.00042575502416576953, '2': 0.0002800057781236899}
- psnr: mean delta VQGAN-VQAE = -0.422244, 95% cluster CI [-0.469243, -0.382878], per-seed {'0': -0.4008138628596964, '1': -0.3935514630338514, '2': -0.47236675011900453}
- full_rmse: mean delta VQGAN-VQAE = 0.00368145, 95% cluster CI [0.00331394, 0.0041861], per-seed {'0': 0.0034125089514418514, '1': 0.0034229847733513456, '2': 0.004208842001389712}
- centered_rmse: mean delta VQGAN-VQAE = 0.00367935, 95% cluster CI [0.0033007, 0.00420291], per-seed {'0': 0.0034104585938621312, '1': 0.003420624969294292, '2': 0.004206971134408379}
- relmeaserr: mean delta VQGAN-VQAE = 1.26589e-09, 95% cluster CI [-3.25195e-10, 2.8716e-09], per-seed {'0': 4.984017920595093e-10, '1': 1.832601959073089e-09, '2': 1.4666701464438615e-09}

## Transfer Ceiling
- all stage0 transfer headroom pass: `True`
- VQGAN teacher LPIPS better seeds: 3/3
- VQGAN teacher KID better seeds: 3/3
- VQGAN teacher RAPSD better seeds: 3/3

## Hash Discipline
- duplicate-clean all seeds: `True`

## Decision
Quality-mode VQGAN improves LPIPS by the preregistered margin with cluster CI excluding zero, RAPSD same direction, measurement consistency passing, and PSNR within 2.5 dB tolerance.
