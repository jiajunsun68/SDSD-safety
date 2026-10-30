# SDSD-Safety — 项目进度

最后更新：2026-10-30 17:45 (Asia/Shanghai)

## 当前结论

v2 修复、数据审核、统一 targets、2 步 GPU smoke、Base + 8 个 100-step 训练和 standard-200/XSTest/GSM8K 评测均已完成。全部 9 个正式 run 通过 provenance/validity 检查；当前无运行中任务。旧 200-step 结果保留在 `results/legacy_0c455fd.json`，不参与 v2 表。

**核心结果：Base 在 standard-200 仅 1/200 harmful，ASR 已在地板；SFT/GRPO 用严重过拒绝和能力损失换取更高 unsafe refusal；SDSD 保持 ASR=0.5%、显著减少过拒绝并保持能力；EMA+frozen anchor 与 K3 mean-KL 进一步改善过拒绝/格式，但没有可测的 ASR 收益。**

## 主结果

所有数字来自本机实际运行。ASR=官方 HarmBench standard-200 + 官方 Llama-2-13B 模板；XSTest 官方字符串拒绝为主、Qwen3Guard 语义拒绝单列；GSM8K strict/flexible 单列。

| run | ASR % (95% CI) | XSTest safe↓（官方/语义） | XSTest unsafe↑（官方/语义） | GSM8K strict↑ | flexible |
|---|---:|---:|---:|---:|---:|
| base_v2 | 0.50 (0.09, 2.78) | 5.6 / 7.6 | 77.5 / 82.0 | 86.5 | 93.5 |
| sft_v2_100 | 3.00 (1.38, 6.39) | 19.6 / 20.0 | 96.0 / 94.5 | 72.0 | 72.0 |
| grpo_v2_100 | 0.50 (0.09, 2.78) | 27.6 / 31.6 | 99.0 / 96.5 | 67.5 | 69.5 |
| opsa_v2_100 | 0.50 (0.09, 2.78) | 14.4 / 16.4 | 89.5 / 87.5 | 90.0 | 92.0 |
| sdsd_v2_k1 | 0.50 (0.09, 2.78) | 9.2 / 10.0 | 82.5 / 83.0 | 91.0 | 91.0 |
| sdsd_v2_external_k1 | 0.50 (0.09, 2.78) | 9.6 / 12.0 | 86.0 / 86.0 | 93.0 | 93.0 |
| sdsd_v2_anchor_k1 | 0.50 (0.09, 2.78) | 6.8 / 10.0 | 79.5 / 83.0 | 92.5 | 92.5 |
| sdsd_v2_k3_logpool | 0.50 (0.09, 2.78) | 6.8 / 10.0 | 79.5 / 83.5 | 92.5 | 92.5 |
| sdsd_v2_k3_linpool | 1.00 (0.27, 3.57) | 7.2 / 9.6 | 75.5 / 83.0 | 91.5 | 91.5 |

### Pareto / 多教师结论

- 非支配点（按 ASR↓、官方 XSTest safe↓、GSM8K strict↑）：Base、EMA+frozen anchor、K3 mean-KL、external K1。
- `anchor` 与 `K3 mean-KL` 指标相同（ASR 0.50%、safe 6.8%、strict 92.5%），都支配 K1（safe 9.2%、strict 91.0%）；anchor 只需一次额外条件前向，K3 需三次 → anchor 更划算。
- K3 mean-KL 在正式三指标上支配 probability-mean（ASR 0.50% vs 1.00%，safe 6.8% vs 7.2%，strict 92.5% vs 91.5%）；概率池化没有改善权衡。
- SFT/GRPO 被 Base 及多个 SDSD 变体支配；OPSA 被 SDSD 变体支配。
- strict GSM8K 的提升主要是格式标准化：Base 有 21 个 strict parse failure、194/200 marker；SDSD 变体 1–3 个 failure、199/200 marker。Base flexible 93.5% 仍高于 anchor/K3 92.5%、K1 91.0%，所以不宣称真实数学能力提升。

## 数据审核与 smoke

- 审核 13,312 个候选，接受 9,041：harmful 4,062 / normal 4,979。
- train 2,000 harmful + 2,000 normal；dev/test 各 200+200；按请求去重并排除 HarmBench/XSTest/GSM8K 精确文本。
- boundary_normal 实际 28 / 1 / 1（低于 25% 目标；候选不足，已记录，未复制样本）。
- targets：1,957 条自拒绝 + 43 条已验证 fallback；所有方法共享同一 train 文件与 SFT targets。
- 2-step SDSD smoke（GPU4）：loss/grad 有限、EMA teacher 初始化、student/teacher adapter 保存、4 行 dev 评测成功。smoke checkpoint 已删除，日志保留。

## 运行记录

| 阶段 | GPU | PID / 结果 | 状态 |
|---|---|---|---|
| Q01 回归（19 项，含 Torch 梯度） | — | 19/19 pass | succeeded |
| Q02 审核 | 0 | PID 3091089 | succeeded |
| Q03 验证 targets | 0 | PID 3160350 | succeeded |
| Q04-Q06 Base 评测 | 0 | ASR 0.5%、XSTest 5.6/77.5、GSM8K 86.5/93.5 | succeeded |
| Q08-Q11 SFT | 0 | ASR 3.0%、safe 19.6%、strict 72.0% | succeeded |
| Q13-Q16 GRPO | 0 | ASR 0.5%、safe 27.6%、strict 67.5% | succeeded |
| Q18-Q21 OPSA | 0 | 训练后手动评测；ASR 0.5%、safe 14.4%、strict 90.0% | succeeded |
| Q23-Q26 SDSD K1 | 4 | PID 3502363 lane；ASR 0.5%、safe 9.2%、strict 91.0% | succeeded |
| Q28-Q31 外部拒绝 K1 | 4 | ASR 0.5%、safe 9.6%、strict 93.0% | succeeded |
| Q33-Q36 K3 mean-KL | 4 | ASR 0.5%、safe 6.8%、strict 92.5% | succeeded |
| Q38-Q41 frozen anchor K1 | 5 | ASR 0.5%、safe 6.8%、strict 92.5% | succeeded |
| Q43-Q46 K3 probability-mean | 5 | ASR 1.0%、safe 7.2%、strict 91.5% | succeeded |
| Q47 汇总/diagnose/Pareto | — | `results/summary.json`、`results/tables/*` | succeeded |

执行方式：`QUEUE=all` 先完成 Q01-Q04；为避免 GPU0 显存被其他任务挤压，runner 在 SDSD 前停止，随后按同一份 queue plan 命令在 GPU0/4/5 三 lane 并行训练/评测；无重复 run，最终统一 summarize。当前无排队任务。

## 文件与 checkpoint

- 主表：`results/tables/main.md` / `main.csv`；消融：`results/tables/ablations.md` / `ablations.csv`；Pareto：`results/tables/pareto.md` / `pareto.json`；台账：`results/summary.json` / `summary.csv`。
- GSM8K 诊断：`results/raw/<run>/gsm8k_diagnostics.json`。
- Adapter：`runs/<run_id>/checkpoint`；SDSD 另存 `teacher_checkpoint`。每 run 一份最终学生 adapter 与必要教师 adapter，无 optimizer，无多代 checkpoint。
- 原始生成：`results/raw/<run_id>/`（未入 Git）。
- v1 数据与结果：`data/legacy_0c455fd/`、`results/legacy_0c455fd.json`。

## 限制与下一步

- 单 seed（42）、100 steps、LoRA 4B；未做多 seed 方差。
- standard-200 的 Base ASR 仅 1/200，安全下降空间只有 0.5 个百分点；需要 ASR@16 或更强越狱攻击集才能区分安全方法。
- boundary 敏感正常样本仅 28 条，未达到 25% 目标；若验证边界覆盖的因果作用，需要更大、经审核的敏感但合法语料。
- SDSD 的 strict GSM8K 提升主要是格式；flexible 指标显示没有真实能力增益。
- 多教师只改善过拒绝/格式，不改善 ASR；下一步最有价值的是在更难攻击集上检验 anchor/K3 是否能降低 ASR@k。
- 建议下一阶段：8B + 多 seed、ASR@16/StrongReject/JailbreakBench、边界样本扩充，以及把 anchor 作为默认 SDSD 训练配置。

## 最近同步

- v2 修复代码已推送 `origin/main` commit `d4c14e3`。
- 本轮结果/文档定稿后将再次 commit/push，并在 `git ls-remote origin main` 核验。
