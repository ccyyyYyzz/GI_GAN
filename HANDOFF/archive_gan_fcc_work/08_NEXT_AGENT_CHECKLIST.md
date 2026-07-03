# Next Agent Checklist

新会话接手时，先做这张清单。不要一上来训练或清理。

## A. 定位

- [ ] 打开 `E:\GAN_FCC_WORK\handoff\00_START_HERE_FOR_NEW_AGENT.md`
- [ ] 读 `01_RESEARCH_STORY.md`
- [ ] 读 `02_THEORY_CORE.md`
- [ ] 读 `07_RED_LINES_AND_WORKING_RULES.md`

## B. 核验当前 repo

```powershell
cd E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code
git status --short --branch
git rev-parse --short HEAD
git remote -v
```

- [ ] 确认分支是 `pub-colab-runner`
- [ ] 记录当前 HEAD
- [ ] 不清理 dirty worktree

## C. 核验关键路径

- [ ] `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.tex`
- [ ] `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.pdf`
- [ ] `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\materials_inventory.md`
- [ ] `E:\GAN_FCC_WORK\experiments\06_gauge_gan_rad5_current_phases69_83`
- [ ] `E:\GAN_FCC_WORK\experiments\04_range_null_counterfactual_barrier_phases48_60`
- [ ] `E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703`
- [ ] `E:\ns_mc_gan_gi_code_fcc_phase1` 仍存在且不要改

## D. 低成本验证

```powershell
python -m pytest tests -q
```

如果失败，先区分：

- 环境缺依赖；
- 测试本身失败；
- 路径假设失效。

不要把环境问题改成科研代码问题。

## E. 做论文任务前

- [ ] 只从 `materials_inventory.md` 抄数字。
- [ ] 缺数字写 `[DATA MISSING]`。
- [ ] 不加 SOTA/diffusion claim。
- [ ] 每个 claim 对应一个 evidence path。

## F. 做实验任务前

- [ ] 写清楚实验二元问题。
- [ ] 写清楚成功/失败判据。
- [ ] 确认 split 不含 test leakage。
- [ ] 确认 checkpoint/init/data order/optimizer/budget。
- [ ] 先短预算 smoke。
- [ ] 保存 per-sample outputs 和 hash。

## G. 做整理任务前

- [ ] 先扫描并写 inventory。
- [ ] 先复制或备份，再移动。
- [ ] frozen 原件只读。
- [ ] 更新 `handoff` 或对应 experiment README，让下一次不用重新猜。

