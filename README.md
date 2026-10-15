# AI4Science R13 — Conformal coverage and width cost / 保形覆盖与宽度代价

研究问题：对 Concrete 抗压强度预测区间施加训练集确定的经验边界后，重新校准能否恢复覆盖？需要付出多少宽度和区间评分代价？配方整组留出后能否保持效果？

Research question: can recalibration recover coverage after clipping Concrete strength intervals to an empirical training-derived corridor, at what width and interval-score cost, and does the effect survive formulation-group holdout?

预注册终版为 [PREREG.md](PREREG.md)（v1.2），与原始冻结稿逐字节相同。冻结时间为 2026-10-15 08:02:54（北京时间）；原始及公开预注册 SHA-256 均为：

`380e7787735461ccf297bc2591cd1679ae269373957013fa0aafbdc58a7c852e`

The final preregistration (v1.2) is byte-identical to the frozen original. [PREREG_SHA256](PREREG_SHA256) verifies all three equivalent copies. [PROVENANCE.json](PROVENANCE.json) distinguishes original hashes from public-copy hashes; [SHA256SUMS](SHA256SUMS) authenticates this package. No experiment was rerun to change the archived judgments.

方法 / Methods: HistGradientBoosting quantile regressors (0.05/0.95), target coverage 90%, fixed parameters and splits. U 为未裁剪区间；N 沿用 U 阈值后裁剪；R 将边界外校准标签的分数设为正无穷，重新定阈值后裁剪。主边界为 [0, 训练标签 P93]。U is unprojected; N clips using the original threshold; R recalibrates with infinite scores for calibration labels outside the corridor, then clips. Legal infinite thresholds return the full corridor. Empty intervals remain in all metrics.

A 使用原始 1030 行及种子 0–5；B 使用去重的 1005 行及种子 100–119；C 按 427 个配方组整体划分，种子同 B；D 使用去重数据与种子 0–5，仅作描述桥接。A uses original rows; B deduplicates; C holds out whole formulation groups; D bridges deduplication and original seeds descriptively. B/C use the frozen shared 10000×20 bootstrap indices. Metrics weight seeds equally.

下表均为主点；差值为 R−N，宽度与 IS 单位 MPa，覆盖单位 %。Full U/N/R tables and paired confidence intervals: [results_R13.md](results_R13.md).

|Branch / 分支|Seeds|N coverage|R coverage|N width|R width|ΔIS|Role / 结论|
|---|---:|---:|---:|---:|---:|---:|---|
|A|6|85.335917|89.857881|18.607291|23.273172|+1.795083|Success / 成功 S_A|
|B|20|84.265873|88.988095|18.803725|26.818947|+5.250701|Success / 成功 S_B|
|C|20|85.660621|88.524072|22.654534|33.816192|+9.342226|Uncertain / 不确定 U_GROUP_BOUNDARY|
|D|6|85.119048|88.492063|18.993021|22.683903|+1.838988|Descriptive only / 仅描述|

**H3 失败 / H3 failed (F_IS_WORSE):** B 的 ΔIS 95% CI 为 [1.652357, 10.207451] MPa，完全高于零，评分越低越好。Coverage recovery carries a positive width cost; interval-score efficiency worsens. Combination SUCCESS consumes A/B only and does not establish scoring improvement.

**C 不确定 / C uncertain:** C−B 的 R 覆盖差 CI 为 [−3.051732, 2.075566] pp，覆盖增益差 CI 为 [−3.137199, −0.651297] pp；两者均跨越预注册的 −2 pp 门槛。C 的 R 有 5/20 个任务、1243/4936 行退回完整走廊。Neither improvement nor deterioration beyond the registered boundary is established; fallback is included, not removed.

**整轮程序完整性带审计保留 / Whole-round procedural integrity carries audit qualification.** [D-01](DEVIATIONS.md): 首次正式尝试因执行沙箱退出而终止，已经产生 A 科学对象；重启时钟违反冻结预注册 §§8.2/11。首次尝试缺少完整 FINISHED/CPU 收尾账本；重启决定前是否有人员或自动化检视 A，现有日志无法确定，也不能据此认定结果选择。T4=PASS 仅覆盖第二次运行（北京时间 10:10:36–10:12:24），不能证明整轮跨尝试资源核算完整。A/B/H3/C 科学判定保留，整轮程序完整性不予通过。

The first attempt produced an A object before termination. Resetting clocks on relaunch breached the preregistration. Its CPU close-out is missing, and pre-relaunch inspection of A is unknown. There is no attestation that A was unread; records neither prove nor exclude a peek, and do not establish result-based selection. The second run's T4 PASS cannot close the cross-attempt ledger. These unresolved limitations survive publication. Second-run wall/CPU: 108.113016/106.059855 seconds; 236 fits and 7344 completed evaluation tasks.

复现 / Reproduction (offline after dependencies are installed): use Python **3.12.3** with [requirements.txt](requirements.txt). Installing dependencies is a separate local setup step; the scripts do not fetch sources or publish anything.

```sh
python -m pip install -r requirements.txt
python -B scripts/verify_bundle.py
python -B scripts/verify_R13_readonly.py
python -B scripts/reproduce.py --output ../r13_reproduction
```

前两项核验分别检查公开包的文件完整性和全部逐行数值、汇总、共享 CI、四对象判定；数值核验只读，不训练、不写回归档，可用 `--output ../verification_public.json` 另存核验报告。独立复现必须使用全新的目录，重新生成公开代码锁和缓存键，再训练全部 236 次并评估 7344 个任务；不修改这里的归档，也不能修复 D-01。The numerical verifier replays archived predictions without training; the reproduction command trains anew in an isolated directory. Original upstream-source/FRP-reference attestations are archived provenance, not freshly reauthenticated by this package.

数据来自 [UCI Concrete](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)；研究参照 [Information 17(10), 946](https://doi.org/10.3390/info17100946)，固定上游提交 `4c5befc6305bdcdda11f4411958c6373f9028f56`。这里的固定数据随机划分只描述划分稳定性，不代表独立实测数据集或未来材料总体；训练分位边界为经验边界，FRP 仅为合成诊断，C 不声称行级交换性。

结构参照 R12：根目录文档、预注册与哈希；scripts/ 核验及复现；data/ 公共来源数据；run/ 汇总、对象、逐行产物及执行代码。隐私与无损压紧记录见 [SANITIZATION.md](SANITIZATION.md) 和 [PRIVACY_SCAN.txt](PRIVACY_SCAN.txt)。First-attempt backups, raw runtime logs, account metadata and installed environments are excluded. No network publication or push was performed.

本公开包自查 / Validation: 全部 7344 项逐行核验通过；在独立目录离线重训 236 次，118 个预测族逐数组完全一致，分支与配对汇总及四对象判定记录完全一致。Evidence: [read-only verification](results/public_readonly_verification.json), [reproduction comparison](results/public_reproduction_check.json). This successful reproduction does not resolve D-01.
