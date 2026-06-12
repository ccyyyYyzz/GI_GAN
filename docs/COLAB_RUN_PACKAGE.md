# Colab Run Package

This package is for running extra Phase 10/11 jobs on Google Colab without colliding with the local Windows run.

Do not run `hadamard10_full_noise001` on Colab while the local machine is already training it. Use Colab for independent outputs:

- `hadamard5_medium`: highest priority Colab job.
- `rademacher10`: control job.
- `scrambled10`: control job.
- `mnist5` / `fashion5`: simple-domain sanity.
- `hadamard5_push`: only after the 5% medium result is close or passing.

## Colab Cells

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Upload or clone this repository so it is available at:

```text
/content/ns_mc_gan_gi
```

Then run:

```bash
%cd /content/ns_mc_gan_gi
!bash scripts/colab_setup.sh
```

Recommended first Colab job:

```bash
!bash scripts/colab_run_recommended.sh hadamard5_medium
```

Other tasks:

```bash
!bash scripts/colab_run_recommended.sh rademacher10
!bash scripts/colab_run_recommended.sh scrambled10
!bash scripts/colab_run_recommended.sh mnist5
!bash scripts/colab_run_recommended.sh fashion5
```

Check status:

```bash
!bash scripts/colab_status.sh configs/colab/hadamard5_medium_noise001_colab.yaml
```

Pack outputs:

```bash
!bash scripts/colab_pack_outputs.sh configs/colab/hadamard5_medium_noise001_colab.yaml
```

The archive is written under:

```text
/content/drive/MyDrive/ns_mc_gan_gi/colab_archives
```

## Resume After Colab Disconnect

Rerun the same task command. `scripts/colab_run_task.sh` automatically resumes from `last.pt` when it exists.

Example:

```bash
!bash scripts/colab_run_recommended.sh hadamard5_medium
```

## Output Paths

Colab configs write to Drive:

```text
/content/drive/MyDrive/ns_mc_gan_gi/data
/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase10_colab
/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase11_colab
```

Each config has a unique output directory, so multiple machines do not overwrite each other.

## Copy Results Back To Windows

After a Colab task completes, download the `.tar.gz` from Drive and extract it into the matching Windows output directory.

For example, `hadamard5_medium_noise001` should be copied into:

```text
E:/ns_mc_gan_gi/outputs_phase10/hadamard5_medium_noise001
```

Then refresh local aggregation:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -m src.aggregate_phase10
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -m src.phase11_attribution
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -m src.aggregate_phase11
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -m src.make_phase11_report
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -m src.export_phase11_paper_assets
```

## Colab Task Map

| Task | Config | Output |
|---|---|---|
| hadamard5_medium | `configs/colab/hadamard5_medium_noise001_colab.yaml` | `outputs_phase10_colab/hadamard5_medium_noise001` |
| rademacher10 | `configs/colab/rademacher10_full_noise001_colab.yaml` | `outputs_phase10_colab/rademacher10_full_noise001` |
| scrambled10 | `configs/colab/scrambled_hadamard10_full_noise001_colab.yaml` | `outputs_phase10_colab/scrambled_hadamard10_full_noise001` |
| mnist5 | `configs/colab/mnist_hadamard5_full_colab.yaml` | `outputs_phase10_colab/mnist_hadamard5_full` |
| fashion5 | `configs/colab/fashion_hadamard5_full_colab.yaml` | `outputs_phase10_colab/fashion_hadamard5_full` |
| hadamard5_push | `configs/colab/hadamard5_push_hq_colab.yaml` | `outputs_phase11_colab/hadamard5_push_hq` |

## Important Rules

- Do not write Colab outputs to `E:/...`.
- Do not write local Windows outputs to `/content/...`.
- Do not claim a Colab run is complete until `eval_metrics.json` exists.
- Do not call a resumed partial Colab run full unless it reaches the target epoch and sample counts in the config.
- Do not use FakeData as a substitute for STL-10/MNIST/FashionMNIST.
