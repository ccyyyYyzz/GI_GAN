$ErrorActionPreference = "Stop"
$envPath = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$configs = @(
  "configs/phase8_hq/mnist_hq_5pct.yaml",
  "configs/phase8_hq/fashion_mnist_hq_5pct.yaml",
  "configs/phase8_hq/cifar10_gray_hq_10pct.yaml"
)
foreach ($cfg in $configs) {
  conda run -p $envPath python -m src.train --config $cfg --device cuda
}
