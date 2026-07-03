# Red Lines And Working Rules

这些规则来自项目历史中踩过的坑。新 agent 应先遵守，再讨论是否改变。

## 文件系统红线

1. 不要移动、删除、编辑：

   `E:\ns_mc_gan_gi_code_fcc_phase1`

   这是 frozen 原始混合目录。只读扫描和复制可以，原地改动不可以。

2. 大规模移动/清理前必须先建 manifest，并确认备份位置。

3. active repo dirty worktree 是已知状态。不要 `git reset --hard`，不要自动清理 untracked。

4. 新代码默认写在：

   `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code`

   新证据默认写在对应实验输出目录，并更新 README/manifest。

## 实验红线

1. 不准用 test split 参与训练或调参。

2. 不准事后修改成功判据来让结果好看。

3. 不准报告没有 per-sample 输出支撑的采样结论。

4. 不准声称 posterior collapse 已解决，除非有：

   - `mean pixel std > 0.01`
   - `P0 variance > 1e-4`
   - `P0/PR variance ratio > 5`
   - per-sample `RelMeasErr < 1e-2`
   - `P0` power spectrum 不是白噪声假阳性

5. 多 seed 实验必须遵守预注册 seed 数，不因为结果暧昧就临时加 seed。

## 论文红线

1. 所有数值必须来自：

   `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\materials_inventory.md`

   或来自明确、可追溯的新结果。缺失写 `[DATA MISSING]`。

2. 不 claim SOTA，不 claim 击败 diffusion。

3. 不把 measurement consistency 写成 semantic correctness。

4. 每个图/表必须能回到数据、脚本、配置或 manifest。

## Compute 红线

1. Colab/GPU 不是默认动作。只有任务明确要求训练时才启动。

2. 训练前必须确认：

   - checkpoint/init；
   - split；
   - data order；
   - optimizer；
   - budget；
   - selection rule；
   - 唯一变量。

3. 训练后必须保存：

   - config；
   - checkpoint hash；
   - split hash；
   - per-sample outputs；
   - metrics；
   - code hash / dirty diff 摘要。

## 解释红线

如果 RelMeasErr 很低，只能说明测量一致；不能推出图像正确。

如果 PSNR 没变但 RelMeasErr 降了，这是 range-null separability 的预期，不是失败。

如果 P0 多样性升高但功率谱像白噪声，这是垃圾多样性，不是真 posterior 成功。

