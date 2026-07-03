# Reproducibility Audit

Generated: 2026-06-14 23:28:55

- Branch target: `pub-colab-runner`; captured in `00_GIT_STATUS.md`.
- Worktree status: dirty/untracked; see `GIT_SAFETY_WARNING.md`. Phase79 did not reset, clean, checkout, or overwrite old artifacts.
- Compile check: `python -m compileall src` in `E:/ns_mc_gan_gi_code` completed with exit code 0. Log: `06_COMPILE_LOG.txt`.
- Full hash table: `06_HASH_TABLE.csv`. Hashes were computed for files up to 25 MB; larger checkpoints/arrays are recorded with size and skip note unless a phase manifest already supplied hash.
- First-paper exact-A: Rad-5/Rad-10 exported tensors are canonical; Scr-5/Scr-10 are regenerated and indirectly validated by exact metric/backprojection reproduction.
- Splits: main train sorted SHA `04ac1e4f5ed20b05d126ed2af41a0a9d7644f357a49c6ae37bc3ec85fbf3f097`; main eval sorted SHA `84c88e09ecdd7584a717f63accbacfecf92501a806c84473adb69811d8c30b1c`.
- GAN train/val/test guard: Phase73/74/75 split manifests report train/val overlap 0 and test overlap 0, with frozen test SHA `27227dcd4f3d28549f981ec449ff8ba34baaebfbcf754091cbc0ddc218d3384d`.
- Colab: no new Colab runs were launched in this Phase79 audit.
- Training: no new training was launched by this Phase79 full-paper audit. Existing Phase78/Phase79 prior outputs include exploratory training and are explicitly marked exploratory/negative where applicable.
