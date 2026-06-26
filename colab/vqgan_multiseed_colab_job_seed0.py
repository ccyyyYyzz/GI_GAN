import importlib
import sys

sys.modules.pop("vqgan_multiseed_colab_job_common", None)
common = importlib.import_module("vqgan_multiseed_colab_job_common")

common.main(seed_id=0)
