# SDSD-Safety — 项目进度

最后更新：2026-10-30 22:55 (Asia/Shanghai)

## 当前阶段：ACL 方法论文 Round 1（无新训练）

- 从 `46a0f88` 继续，v2 的 9 个正式 checkpoint、原始输出和 provenance 全部复用；不重训、不删除现有 checkpoint。
- 回归 26/26 通过（含 Torch 梯度、SIRL 全词表熵奖励、ASR 组合估计、配对 bootstrap）。
- 已写并提交前先固定协议：`docs/gsm8k_protocol.md`（OpenAI GSM8K commit `2909d34`，exact string）与 `docs/asr_protocol.md`（固定 16 样本、ASR@1/4/16、未知标签上下界、request-level paired bootstrap）。
- GSM8K 已对 v2 的 9 份保存输出重评，未重新调用模型。公开 exact scorer 主列与旧 strict/flexible 诊断差异极大：多数模型写作 `####3` 而官方 extractor 要求 `#### `，因此 exact 分数衡量格式服从性，不能单独当作数学能力。旧 summary 已归档在 `results/archive/gsm8k_openai_rescore_v1/`。
- 新采样代码的 4-sample/2-behavior 与 16-sample/3-behavior GPU smoke 均成功；behavior 与 attack_prompt 分列，分类器只读取原 behavior，manifest 含训练身份、评估身份、代码/提示/解码/judge/输出哈希。
- 代码已扩展 SIRL（旧策略全词表熵的负均值、detach、group advantage、clip、reference KL，无 guard margin）和 EMA-demo + frozen-plain 条件控制；Round 2 尚未训练。
- Round 1 队列已 plan（Q01-Q09），但当前没有进程在运行。下一步在 GPU0/4 两 lane 实际执行 5 个冻结模型的 200×16 采样与统一分类。任何计划状态不等于运行结果。

## v2 当前结论（2026-10-30）

v2 修复、数据审核、统一 targets、2 步 GPU smoke、Base + 8 个 100-step 训练和 standard-200/XSTest/GSM8K 评测均已完成。全部 9 个正式 run 通过 provenance/validity 检查；当前无运行中任务。旧 200-step 结果保留在 `results/legacy_0c455fd.json`，不参与 v2 表。

**核心结果：Base 在 standard-200 仅 1/200 harmful，greedy ASR 已在地板；SFT/GRPO 用严重过拒绝和 flexible 能力损失换取更高 unsafe refusal；SDSD 系列官方过拒绝较低，flexible GSM8K 接近 Base，但公开 exact 主列显示所有安全适配仍有格式服从成本。该 scorer 冲突必须在论文中同时报告；旧 strict 主列 Pareto 已作废。**

## 主结果（v2 greedy；GSM8K 主列已切换公开 exact）

所有数字来自本机实际运行。ASR=官方 HarmBench standard-200 + 官方 Llama-2-13B 模板；XSTest 官方字符串拒绝为主、Qwen3Guard 语义拒绝单列；GSM8K 主列为 `openai-grade-school-math@2909d34` exact，strict/flexible 仅作格式诊断。

| run | ASR greedy % (95% CI) | XSTest safe↓（官方/语义） | XSTest unsafe↑（官方/语义） | GSM8K exact↑ | strict 诊断 | flexible 诊断 |
|---|---:|---:|---:|---:|---:|---:|
| base_v2 | 0.50 (0.09, 2.78) | 5.6 / 7.6 | 77.5 / 82.0 | 43.0 | 86.5 | 93.5 |
| sft_v2_100 | 3.00 (1.38, 6.39) | 19.6 / 20.0 | 96.0 / 94.5 | 0.0 | 72.0 | 72.0 |
| grpo_v2_100 | 0.50 (0.09, 2.78) | 27.6 / 31.6 | 99.0 / 96.5 | 42.5 | 67.5 | 69.5 |
| opsa_v2_100 | 0.50 (0.09, 2.78) | 14.4 / 16.4 | 89.5 / 87.5 | 17.5 | 90.0 | 92.0 |
| sdsd_v2_k1 | 0.50 (0.09, 2.78) | 9.2 / 10.0 | 82.5 / 83.0 | 3.0 | 91.0 | 91.0 |
| sdsd_v2_external_k1 | 0.50 (0.09, 2.78) | 9.6 / 12.0 | 86.0 / 86.0 | 0.5 | 93.0 | 93.0 |
| sdsd_v2_anchor_k1 | 0.50 (0.09, 2.78) | 6.8 / 10.0 | 79.5 / 83.0 | 3.5 | 92.5 | 92.5 |
| sdsd_v2_k3_logpool | 0.50 (0.09, 2.78) | 6.8 / 10.0 | 79.5 / 83.5 | 1.0 | 92.5 | 92.5 |
| sdsd_v2_k3_linpool | 1.00 (0.27, 3.57) | 7.2 / 9.6 | 75.5 / 83.0 | 0.0 | 91.5 | 91.5 |

### 解释（待 Round 1 确认）

- 公开 exact 主列与 flexible 诊断的 40–90 个百分点差距来自答案格式：模型常写 `####3`，而官方 extractor 只接受严格 `#### ` 形式；因此 exact 衡量格式服从性。任何安全适配的能力结论必须同时报告 diagnostics，不能只引用 exact。
- 旧 v2 结论“SDSD 保持能力、降低过拒绝”只在 flexible/strict 诊断下成立；在公开 exact 主列下 Base 仍最高。该冲突是当前最重要的可复现发现之一，必须在新论文中如实呈现，不能静默选择有利 scorer。
- greedy ASR 全在 0.5–3% 区间，Base 仅 1/200，无法区分方法；这正是 Round 1 固定 16-sample ASR@k 要解决的测量问题。
- 旧 Pareto 基于 strict 主列，已被公开 scorer 重评分 supersede；Round 1/2 完成后用 exact + flexible 双列重算并明确主分析。

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

- 主表：`results/tables/main.md` / `main.csv`；消融：`results/tables/ablations.md` / `ablations.csv`；采样表：`results/tables/asr_sampling.md` / `asr_sampling.csv`（Round 1 后）；台账：`results/summary.json` / `summary.csv`。旧 `results/tables/pareto.*` 基于已作废的 strict 主列，不作为当前结论。
- GSM8K 诊断：`results/raw/<run>/gsm8k_diagnostics.json`。
- Adapter：`runs/<run_id>/checkpoint`；SDSD 另存 `teacher_checkpoint`。每 run 一份最终学生 adapter 与必要教师 adapter，无 optimizer，无多代 checkpoint。
- 原始生成：`results/raw/<run_id>/`（未入 Git）。
- v1 数据与结果：`data/legacy_0c455fd/`、`results/legacy_0c455fd.json`。

## 限制与下一步

- 单 seed（42）、100 steps、LoRA 4B；未做多 seed 方差。
- standard-200 的 Base ASR 仅 1/200，安全下降空间只有 0.5 个百分点；需要 ASR@16 或更强越狱攻击集才能区分安全方法。
- boundary 敏感正常样本仅 28 条，未达到 25% 目标；若验证边界覆盖的因果作用，需要更大、经审核的敏感但合法语料。
- 公开 exact 与 strict/flexible 主列冲突：模型常不写官方要求的 `#### ` 空格，exact 主要衡量格式；flexible 下 SDSD 接近 Base，但公开 exact 下明显更低。论文必须双列报告，不能只选有利 scorer。
- 多教师只改善过拒绝/格式，不改善 ASR；下一步最有价值的是在更难攻击集上检验 anchor/K3 是否能降低 ASR@k。
- 建议下一阶段：8B + 多 seed、ASR@16/StrongReject/JailbreakBench、边界样本扩充，以及把 anchor 作为默认 SDSD 训练配置。

## 最近同步

- 2026-10-30 17:50 CST：v2 结果/文档已推送 `origin/main`，结果 commit `ffea82c`（`git ls-remote origin main` 已核验一致）。v2 代码修复 commit 为 `d4c14e3`。
- 之后仅记录推送状态的 commit 不含新的实验数字。

<!-- QUEUE:START -->
更新：2026-10-30 22:54:34 CST；队列状态：**planned**；GPU：0

| ID | 任务 | 状态 | PID | Exit | 最近输出 / 原因 |
|---|---|---|---|---|---|
| Q01 | Run regression and Torch gradient tests | planned | — | — | OK |
| Q02 | Generate 16-sample HarmBench for base_v2 | planned | — | — | } |
| Q03 | Generate 16-sample HarmBench for opsa_v2_100_seed42 | planned | — | — | } |
| Q04 | Generate 16-sample HarmBench for sdsd_v2_k1_100_seed42 | planned | — | — | [base_v2] gsm8k: status=completed n=200 valid=True |
| Q05 | Generate 16-sample HarmBench for sdsd_v2_anchor_k1_100_seed42 | planned | — | — |  |
| Q06 | Generate 16-sample HarmBench for sdsd_v2_external_k1_100_seed42 | planned | — | — |  |
| Q07 | Judge repeated samples with official classifier | planned | — | — |  |
| Q08 | Analyze paired ASR@k endpoints | planned | — | — |  |
| Q09 | Refresh validated tables and ledger | planned | — | — |  |
<!-- QUEUE:END -->
