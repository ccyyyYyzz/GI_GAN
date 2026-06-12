# Colab Local Output Mode

Use this mode when Google Drive cannot be mounted or the mounted Drive account is out of space.

Outputs are written to the Colab runtime under:

```text
/content/ns_mc_gan_gi_outputs
```

Archives are written to:

```text
/content/ns_mc_gan_gi_archives
```

These files are temporary. Download each archive before the runtime disconnects.

## Setup

```bash
%cd /content/extracted_ns_mc_gan
!bash scripts/colab_setup_local.sh
```

## Run Tasks

```bash
!bash scripts/colab_run_local_task.sh mnist5
!bash scripts/colab_run_local_task.sh fashion5
```

Other valid tasks:

```text
hadamard5_medium
rademacher10
scrambled10
mnist5
fashion5
hadamard5_push
```

## Pack And Download

```bash
!bash scripts/colab_pack_local_outputs.sh mnist5
!bash scripts/colab_pack_local_outputs.sh fashion5
```

Then download the `.tar.gz` files from:

```text
/content/ns_mc_gan_gi_archives
```

Or use:

```python
from google.colab import files
files.download('/content/ns_mc_gan_gi_archives/<archive-name>.tar.gz')
```
