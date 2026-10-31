# SDSD-Safety — 项目进度

最后更新：2026-10-31 17:40 (Asia/Shanghai)

## 当前阶段：机制发现与修复完成；anchor 角色扮演越狱脆弱性成立，plain-frozen 修复有效，强 plain-prompt KL 对照无效；独立攻击族与跨家族给出清晰边界，论文材料更新中

- 从 `46a0f88` 继续，v2 的 9 个正式 checkpoint、原始输出和 provenance 全部复用；不重训、不删除现有 checkpoint。
- 回归 26/26 通过（含 Torch 梯度、SIRL 全词表熵奖励、ASR 组合估计、配对 bootstrap）。
- 已写并提交前先固定协议：`docs/gsm8k_protocol.md`（OpenAI GSM8K commit `2909d34`，exact string）与 `docs/asr_protocol.md`（固定 16 样本、ASR@1/4/16、未知标签上下界、request-level paired bootstrap）。
- GSM8K 已对 v2 的 9 份保存输出重评，未重新调用模型。公开 exact scorer 主列与旧 strict/flexible 诊断差异极大：多数模型写作 `####3` 而官方 extractor 要求 `#### `，因此 exact 分数衡量格式服从性，不能单独当作数学能力。旧 summary 已归档在 `results/archive/gsm8k_openai_rescore_v1/`。
- 新采样代码的 4-sample/2-behavior 与 16-sample/3-behavior GPU smoke 均成功；behavior 与 attack_prompt 分列，分类器只读取原 behavior，manifest 含训练身份、评估身份、代码/提示/解码/judge/输出哈希。
- 代码已扩展 SIRL（旧策略全词表熵的负均值、detach、group advantage、clip、reference KL，无 guard margin）和 EMA-demo + frozen-plain 条件控制；Round 2 尚未训练。
- Round 1 已完成：四个 16-sample 生成（base/K1/anchor/external）于 23:29–23:42 完成；OPSA 16-sample 生成于 23:41:59 完成（200×16，695.7s）；五个 run 的官方 HarmBench classifier 判分于 23:36–23:53 完成（GPU6 主判分 PID 1691；GPU0 OPSA 判分 PID 32774），unknown=0。`analyze_asr` 与 `evaluate --phase summarize` 已刷新配对表、曲线和规范表。

| run | 生成 | 判分 ASR@1 / @4 / @16 (%) | 状态 |
|---|---|---:|---|
| base_v2 | 23:36（708.9s） | 0.50 / 0.50 / 0.50 | completed |
| opsa_v2_100_seed42 | 23:41:59（695.7s） | 0.47 / 0.50 / 0.50 | completed |
| sdsd_v2_k1_100_seed42 | 23:42（635.2s） | 0.50 / 0.50 / 0.50 | completed |
| sdsd_v2_external_k1_100_seed42 | 23:52（731.3s） | 0.50 / 0.50 / 0.50 | completed |
| sdsd_v2_anchor_k1_100_seed42 | 23:48（792.8s） | 0.53 / 0.62 / 1.00 | completed |

**Round 1 结论：** 固定直接请求的 ASR@k 全部贴地板，Base/K1/external/OPSA 在 @1/@4/@16 均为 0.47–0.50%；anchor 在 @16 为 1.00%（比 Base 多 1 个 only-A harmful 行为，Δ=+0.50pp，95% CI [0.00,1.50]，不显著）。**完整方法没有降低可采样的有害输出概率，直接请求协议缺乏区分度。** 逐请求重叠：anchor−base 为 only A=1、both=1、neither=198；其余比较 both=1、only=0。截断检查：同一 192-token 预算下 length-stop 比例 base 35.2%、anchor 35.8%、K1 11.8%、external 4.6%、OPSA 3.9%——方法间差异明显，但全部方法使用同一冻结预算；论文需报告该协议限制。

**分支判定（按 prompt 第七节 C 与第六节）：** anchor 尚未降低有害采样概率；本地 HarmBench 无预计算攻击缓存（`data/` 只有 behavior CSV 与 classifier 校验集，无 test_cases），但有本地源模型（Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct 等）。下一项应先在开发样本上冻结一个官方固定攻击协议（优先 HumanJailbreaks 固定提示或 GCG-Transfer），对所有方法使用同一攻击提示、源模型/文件哈希与输出预算，判分与 @k 单列；直接请求 @k 结果保留，不冒充攻击结果。

- 2026-10-31 00:03：Q10 启动。新增 `sdsd_safety/harmbench_attack.py`，固定攻击协议 `sdsd-safety-hb-humanjb-fixed5-dev-v1`：官方 HumanJailbreaks prompt index 5（源文件 `baselines/human_jailbreaks/jailbreaks.py`，sha256 `62e53558...`，prompt sha256 `af5d1a...`），HarmBench standard-validation 41 behaviors（`harmbench_behaviors_text_val.csv`，sha256 记录于 manifest），attack_prompt = jailbreak + blank line + behavior，与官方 baseline 构造一致；temperature 1、top-p 1、top-k 0、max 320 new tokens；全部 prompt 250–271 tokens（≤512）。分类器仍只读原 behavior；行为/攻击提示/回答三字段分列。输出独立目录 `results/raw_attack/humanjb-fixed5-v1/<run>/`，不覆盖 asr-sample-v3。GPU6 PID 68088，日志 `logs/attack_humanjb_fixed5.log`。**Q10 结果（00:37:55 完成）：五个方法 ASR@1/@4/@16 全部为 0.00%，unknown=0。** 该固定 HumanJailbreaks prompt 对 Qwen3 系列全部被拒绝（抽查回答为“I cannot play the role of FreeSpeechGPT…”），不具区分度；作为有效负结果保留，不冒充安全收益。输出 `results/raw_attack/humanjb-fixed5-v1/`、`results/tables/attack_humanjb_fixed5.md`。

- 2026-10-31 00:40：Q11 dev 协议选择启动。预注册规则：按官方 `jailbreaks.py` 列表顺序测试 index 0–7 共 8 个固定 prompt（不改写、不按回答内容挑选），每个在 Base 上跑 HarmBench standard-val 41 behaviors × 4 samples（temperature 1, top-p 1, top-k 0, max 320），取 ASR@1 最高者（并列取最小 index）冻结，随后对五个方法跑 16 samples。GPU6 PID 131531，日志 `logs/attack_sweep.log`，输出 `results/raw_attack/humanjb-sweep4-v1/idx{0..7}/`。**状态：RUNNING**。

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
更新：2026-10-31 01:50 CST；队列状态：**Round 1 completed; fixed-attack dev found a large anchor vulnerability; held-out confirmation running**；GPU：0

| ID | 任务 | 状态 | PID | Exit | 最近输出 / 原因 |
|---|---|---|---|---|---|
| Q01 | Run regression and Torch gradient tests | completed | — | 0 | 26/26 pass (23:28) |
| Q02 | Generate 16-sample HarmBench for base_v2 | completed | 4136540 | 0 | 200×16, 708.9s, log `logs/round1_hb16_base_v2.log` |
| Q03 | Generate 16-sample HarmBench for opsa_v2_100_seed42 | completed | 3102 | 0 | 200×16, 695.7s, log `logs/round1_hb16_opsa.log` |
| Q04 | Generate 16-sample HarmBench for sdsd_v2_k1_100_seed42 | completed | 4136616 | 0 | 200×16, 635.2s |
| Q05 | Generate 16-sample HarmBench for sdsd_v2_anchor_k1_100_seed42 | completed | 4136704 | 0 | 200×16, 792.8s |
| Q06 | Generate 16-sample HarmBench for sdsd_v2_external_k1_100_seed42 | completed | 4136777 | 0 | 200×16, 731.3s |
| Q07 | Judge repeated samples with official classifier | completed | 1691 / 32774 | 0 | five runs, unknown=0, logs `logs/round1_judge_4runs.log`, `logs/round1_judge_opsa.log` |
| Q08 | Analyze paired ASR@k endpoints | completed | — | 0 | `results/analysis/asr_round1.json`, `results/tables/asr_round1_paired.*`, `results/figures/asr_round1_asr_vs_k.png` |
| Q09 | Refresh validated tables and ledger | completed | — | 0 | `results/tables/asr_sampling.*`, `results/summary.json` (18 runs) |
| Q10 | Fixed-attack dev protocol: HumanJailbreaks index 5 × HarmBench standard-val 41 behaviors × 16 samples × 5 runs | completed | 68088 | 0 | all five ASR@1/@4/@16 = 0.00%, unknown=0; log `logs/attack_humanjb_fixed5.log`; table `results/tables/attack_humanjb_fixed5.md` |
| Q11 | Dev protocol selection: official HumanJailbreaks indices 0–7 on Base, 41 val behaviors × 4 samples | completed | 131531 | 0 | winner idx6 (ASR@1 2.44%, @4 2.44%); idx2 excluded (>512 tokens); log `logs/attack_sweep.log`; output `results/raw_attack/humanjb-sweep4-v1/` |
| Q12 | Fixed attack idx6 × HarmBench standard-val 41 behaviors × 16 samples × 5 runs | completed | 165690 / 214096 | 0 | anchor ASR@16 48.78% vs base 2.44%; judge retried on GPU0 after GPU6 tenant OOM; table `results/tables/attack_humanjb_idx6_dev.md` |
| Q13 | Held-out confirmation: idx6 × HarmBench text-test 159 standard behaviors × 16 samples × 5 runs | completed | 248058 / 422237 | 0 | anchor ASR@16 57.23% vs base 2.52% (Δ+54.72pp CI [+47.17,+62.26]); 91/159 behaviors harmful; log `logs/attack_idx6_test159_judge.log`; table `results/tables/attack_humanjb_idx6_test159.md` |
| Q14 | Decide next: robustness repair vs trade-off paper; no automatic training/attack | completed | — | 0 | chose mechanism-guided repair: P1 teacher-context probe + P2 scale pilot + R1 plain-frozen; results below |
| Q15 | P1 teacher x context x prefix probe | completed | — | 0 | `results/analysis/teacher_context_probe.json`, `results/tables/teacher_context_probe.md` |
| Q16 | P2 LoRA scale pilot (K1/anchor, s=0/0.25/0.5/1.0) | completed | 1262685 | 0 | dose threshold: anchor s<=0.25 Base-level ASR, s=1 35.37/41.46%; normal dev refusal 0% at s>=0.25; `results/tables/scale_pilot.md` |
| Q17 | R1 plain-frozen training (frozen_teacher_demo=false) | completed | 1248532 | 0 | 100/100 steps, dev normal_refusal 2.0%; `runs/sdsd_v3_anchor_plainfrozen_100_seed42/` |
| Q18 | R1 evaluation: standard200 + XSTest + GSM8K | completed | 1378807 | 0 | HB greedy 0.5%; XSTest 5.6/7.2 safe, 79.0 unsafe; GSM lm-eval 90.5 |
| Q19 | R1 fixed idx6 test159 x16 attack | completed | 1362510 | 0 | ASR@1/@4/@16 = 1.26/2.31/3.14% vs anchor 57.23%; unknown=0 |
| Q20 | R2 K1 + ref_kl 0.3 training (pre-registered branch 1) | completed | 1411299 | 0 | 100/100 steps; dev normal_refusal 1.5%; `runs/sdsd_v3_k1_refkl03_100_seed42/` |
| Q21 | R2 standard evaluation | completed | 1508692 | 0 | HB greedy 0.5%; XSTest 6.4/8.0 safe, 79.0/83.5 unsafe; GSM lm-eval 90.5 |
| Q22 | R2 fixed idx6 test159 x16 attack | running | 1508693 | — | GPU0; generation started 15:20; judge and table refresh follow |
| Q23 | Independent attack family idx8 test159 x16 | completed | 1422952/1423077/1423258 | 0 | Base/anchor/R1 all 0.00% at @1/@4/@16; valid negative, vulnerability bounded to role-play family |
| Q24 | Qwen2.5-7B Base standard evaluation (cross-family baseline) | completed | 1418348 | 0 | HB direct ASR 6.0%; XSTest 4.0/6.0 safe, 67.0/82.5 unsafe; GSM lm-eval 94.5 |
| Q25 | Qwen2.5 anchor training (frozen_teacher_demo=true) | completed | 1531846 | 0 | 100/100 steps; dev normal_refusal 0.5%; `runs/sdsd_v3_qwen25_anchor_100_seed42/` |
| Q26 | Qwen2.5 plain-frozen training (frozen_teacher_demo=false) | completed | 1537490 | 0 | 100/100 steps; dev normal_refusal 1.0%; `runs/sdsd_v3_qwen25_plainfrozen_100_seed42/` |
| Q27 | Qwen2.5 Base fixed idx6 test159 x16 attack | completed | 1552093 | 0 | ASR@1/@4/@16 = 93.12/99.87/100%; length_hit 29.2%; attack at ceiling |
| Q28 | Qwen2.5 anchor/plain-frozen standard evaluation | completed | 1620916/1616750 | 0 | anchor HB 10.5%, GSM 91.0, semantic XSTest 7.2/82.5; pf HB 12.5%, GSM 92.0, 5.6/81.5 |
| Q29 | Qwen2.5 anchor fixed idx6 test159 x16 attack | completed | 1637681 | 0 | ASR 86.71/96.94/98.74% (at ceiling, below Base) |
| Q30 | Qwen2.5 plain-frozen fixed idx6 attack | completed | 1634649 | 0 | ASR 93.20/99.52/100%; all three cross-family models at ceiling |
| Q31 | Paper materials update (research/method/outline) | completed | — | 0 | `docs/research.md` section 8, `docs/method.md` section 9, `docs/paper_outline.md` |
| Q32 | P2 valuable scale points promoted to 16 samples | completed | 1746868/1746869 | 0 | anchor s=0.25 2.59/3.05/4.88; s=0.5 6.55/11.30/14.63; paired CIs vs Base |
| Q33 | Commit compact code/results/docs and push | completed | — | 0 | pushed `origin/main` at d9227f8; no raw generations or weights |
<!-- QUEUE:END -->

- 2026-10-31 17:55：本轮机制/修复/独立攻击/跨家族结果与论文材料已推送 `origin/main`，结果 commit `d9227f8`（前序代码/探针 commit `1d066ed`）；`git ls-remote origin main` 已核验一致。原始生成、日志、checkpoint/权重与环境目录未提交。

- 2026-10-31 01:00：Q11 完成。8 个固定 prompt 的 Base dev 结果（41 val behaviors × 4 samples；index 2 因 prompt >512 tokens 不合法，未截断）：idx0 0/0；idx1 ASR@1=0.61%、@4=2.44%；idx2 超长排除；idx3/4/5/7 均为 0%；**idx6 ASR@1=2.44%、@4=2.44%（最高 @1，按预注册规则胜出）**。全部输出在 `results/raw_attack/humanjb-sweep4-v1/idx{0..7}/`，日志 `logs/attack_sweep.log`。冻结选择：**HumanJailbreaks index 6**（“Buer rules”角色扮演 prompt；源文件 sha256 62e53558...，prompt 文本与哈希记录在 manifest）。
- 2026-10-31 01:02：Q12 启动。用冻结的 idx6 对五个方法跑 HarmBench standard-val 41 behaviors × 16 samples（temp 1, top-p 1, top-k 0, max 320），协议 `sdsd-safety-hb-humanjb-idx6-dev-v1`，输出 `results/raw_attack/humanjb-idx6-v1/`；随后判分与配对分析，表 `results/tables/attack_humanjb_idx6_dev.md`。GPU6 PID 165690，日志 `logs/attack_idx6.log`。**状态：RUNNING**。

- 2026-10-31 01:45：**Q12 完成（冻结 idx6 固定攻击，HarmBench standard-val 41 behaviors × 16 samples）**。GPU6 生成（base 155.8s、opsa 173.5s、k1 228.4s、anchor 429.1s、external 279.5s），判分初次在 GPU6 因其他租户占用 50GB 发生 OOM（真实资源错误），按最小修复仅在 GPU0 重跑 judge（未重生成，无新训练），PID 214096，01:31–01:45。

| run | ASR@1 % | ASR@4 % | ASR@16 % |
|---|---:|---:|---:|
| base_v2 | 1.83 | 2.44 | 2.44 |
| opsa_v2_100 | 2.59 | 3.92 | 4.88 |
| sdsd_v2_k1 | 6.71 | 10.93 | 14.63 |
| sdsd_v2_external_k1 | 16.62 | 23.00 | 26.83 |
| **sdsd_v2_anchor_k1** | **34.91** | **43.16** | **48.78** |

配对（family 级/请求级 95% CI，10,000 reps）：anchor−base @1 +33.08pp [+20.73,+46.19]、@4 +40.72pp [+26.75,+55.22]、@16 +46.34pp [+31.71,+60.98]，均排除 0；K1−base @16 +12.20pp [+2.44,+21.95]；external−base @16 +24.39pp [+12.20,+39.02]；OPSA−base 不显著。Anchor 的 harmful-count 分布：21/41 behaviors 全 0，20/41 至少 1 次有害，其中 7 个 behavior 16/16 全部有害；按语义类别 anchor 有害率为 chemical_biological 32.6%、cybercrime 25.9%、harassment 33.3%、harmful 43.8%、illegal 46.0%、misinformation 25.0%（Base 仅 harassment 25%，其余 0）。抽查回答为 docile roleplay 下的完整有害指令（例如在 “[Buer]” 角色内给出 safrole 提取步骤）；攻击文本未出现在训练数据（grep “Buer”/“FreeSpeechGPT” 均为 0）。**结论：完整 anchor 方法在固定人类越狱协议下显著、广泛地增加了有害输出概率；它降低了过拒绝，却以越狱鲁棒性为代价。原“新增安全收益”假设被否定，论文主张必须据此调整，不能以直接请求地板值掩盖。**

- 2026-10-31 01:50：Q13 启动：held-out 确认。攻击 prompt 仅在 val 41 上选择；用同一冻结 idx6 在 HarmBench text_test 的 159 个 standard behaviors 上对五个 checkpoint 跑 16 samples（协议 `sdsd-safety-hb-humanjb-idx6-test159-v1`；输出 `results/raw_attack/humanjb-idx6-test159-v1/`，表 `results/tables/attack_humanjb_idx6_test159.md`），不重新选择 prompt、不改解码预算。GPU0 启动，日志 `logs/attack_idx6_test159.log`。**状态：RUNNING**。

- 2026-10-31 03:42：Q13 生成完成（159 test standard behaviors × 16 samples × 5 runs；base 659.2s、opsa 1216.9s、k1 1249.8s、anchor 2336.1s、external 1195.3s；length_hit_rate 0.6–9.5%）。首次 judge 在 `judge()` 中触发我自己参数化遗漏的硬编码检查（`expected 41 behaviors`；test159 为 159），属真实代码 bug；最小修复为改用全局 `EXPECTED_BEHAVIORS`（未改生成/协议/数据），仅重跑 judge+analyze，不重生成。GPU0 PID 422234，日志 `logs/attack_idx6_test159_judge.log`。**Q13 状态：RUNNING（judge）**。

- 2026-10-31 04:17：**Q13 完成（held-out 确认）**。judge 修复后 GPU0 PID 422237 于 03:42–04:17 完成五个 run 判分（159 behaviors × 16 samples，unknown=0）；未重新生成、未训练。

| run | ASR@1 % | ASR@4 % | ASR@16 % |
|---|---:|---:|---:|
| base_v2 | 0.94 | 1.50 | 2.52 |
| opsa_v2_100 | 2.75 | 5.29 | 10.69 |
| sdsd_v2_k1 | 7.90 | 12.04 | 15.09 |
| sdsd_v2_external_k1 | 20.83 | 28.90 | 32.70 |
| **sdsd_v2_anchor_k1** | **37.89** | **49.91** | **57.23** |

配对（请求级 95% CI，10,000 reps）：anchor−base @1 +36.95 [+30.74,+43.32]、@4 +48.41 [+41.12,+55.71]、@16 **+54.72 [+47.17,+62.26]**；K1−base @16 +12.58 [+7.55,+18.24]；external−base @16 +30.19 [+23.27,+37.11]；OPSA−base @16 +8.18 [+4.40,+12.58]；全部排除 0。Anchor 91/159 behaviors 至少一次有害；类别有害率 chemical 10.5%、cyber 32.6%、harassment 35.9%、harmful 36.8%、illegal 46.9%、misinformation 49.8%。**dev 的 anchor 越狱脆弱性在未参与 prompt 选择的 159 个 test behaviors 上确认且更强（dev @16 48.78% → test 57.23%）；直接请求上的低 ASR 掩盖了固定人类越狱下的严重不鲁棒。** 输出 `results/raw_attack/humanjb-idx6-test159-v1/`，表 `results/tables/attack_humanjb_idx6_test159.md`，分析 `results/analysis/attack_humanjb_idx6_test159.json`。

**研究判断更新：** 原“anchor 新增安全收益”假设被 held-out 证据否定；当前可支持的贡献是“SDSD/anchor 降低过拒绝与保留 flexible 能力，但以显著增加固定越狱脆弱性为代价，且越狱脆弱性随 anchor 权重/示范条件增强”。论文不能声称安全收益；下一轮（Q14，未启动）二选一：(a) 针对 roleplay/越狱的鲁棒性修复数据或 anchor 正则，再走新确认；(b) 将“安全适配的效率-鲁棒性权衡”作为主论文发现并补充机制/外部验证。固定 HumanJailbreaks 只是单一 prompt、单 seed 的 fixed attack，非 transfer/adaptive；论文需按此限定。

## I. 下一阶段：anchor 失效机制与机制导出的修复（起点 04b1e91，2026-10-31）

研究问题：冻结分支为何放大安全损伤——被冻结的条件分布、学生访问的前缀，还是更新强度？主候选：示范筛选在完整回答层面，蒸馏传递的是给定上下文与学生前缀的 token 分布；冻结参数不自动保留原始上下文下的安全行为。

已完成的三项小修：
- **A（队列 adapter 路径）**：`queue.py` 新增 `expected_adapter_path` / `adapter_argument`；只有显式 Base 使用 `none`，训练 run 在 plan 时绑定预期 checkpoint；`load_model_for_eval` 对缺失目录/训练元数据直接报错，不再静默评估 Base。新增回归测试（新 run 未训练时仍绑定预期路径；acl-round2 不出现 `:none`）。
- **B（攻击参数化）**：`harmbench_attack.py` 支持 `--run id:adapter`、`--jailbreak-index`、`--behaviors-csv`、`--expected-behaviors`、`--protocol-id`、`--attack-dir`、`--table-name`、`--report-name`、`--adapter-scale`；每次输出 `*_per_request.csv`（run_id, behavior_id, n_samples, harmful_count, unknown_count）；manifest 记录 scale、命令与源/提示/权重/judge 身份。
- **C（统一能力计分）**：新增 lm-evaluation-harness `gsm8k` flexible-extract（task v3.0，harness commit `d6de8164...`，regex `(-?[$0-9.,]{2,})|(-?[0-9]+)`，group_select=-1，regexes_to_ignore 与 exact_match 语义逐项复现）。对 9 个 v2 run 的已保存回答统一重评分，未调用模型；旧 OpenAI exact 保留为格式诊断并记录 supersedes。**新主列 GSM8K lm-eval flexible：base 91.0、SFT 90.5、GRPO 91.5、OPSA 91.0、SDSD K1 88.5、external 90.0、anchor 91.0、K3 mean-KL 90.0、K3 prob-mean 89.0**；旧 exact 列（0–43%）只反映 `#### ` 格式服从性。

运行中：
- **R1**：`sdsd_v3_anchor_plainfrozen_100_seed42` 从 Base 重训（frozen_teacher_weight=0.25, frozen_teacher_demo=false, lr 5e-5, ref_kl 0.03, 100 steps, seed 42）；GPU5 PID 1248532，日志 `logs/r1_plainfrozen.log`，13:11 约 24/100 步。2-step GPU smoke 已通过（loss/grad 有限、teacher_context=`ema_demo_plus_frozen_plain_user`、adapter 保存），smoke run 已删除。
- **P2**：K1/anchor 的 LoRA scale 0.25/0.5/1.0（s=0 共享 Base）在 idx6 val41 × 4 samples 上做 pilot，并测 100 条 normal dev 的 greedy 字符串拒绝率；GPU7 PID 1262685，日志 `logs/scale_pilot.log`，base 点已完成（ASR@1=@4=2.44%，normal refusal 1%），其余点运行中。输出 `results/raw_attack/scale-pilot-v1/`、`results/tables/scale_pilot.md`。
- **P1**：已实现于 `analyze_mechanism.py --mode teacher_context`（Base / EMA-K1 / EMA-anchor × plain/demo × no-prefix/short-compliant-prefix；固定 41 val attack + 20 normal 请求；固定 continuation 来自已有 anchor 攻击回答/正常 dev 回答；safe demo 使用独立的 `pku-train-000087` 回答，不复用 REFUSAL_PREFIX 作为评分对象）；尚未运行，待 P2 释放 GPU7。

下一步：R1 完成后跑标准 HarmBench/XSTest/GSM8K + idx6 攻击评估；P2 结果决定“更新幅度”分支；P1 区分上下文/参数/前缀；据此选择一个 R2 训练干预（强 reference KL、低 lr、plain-frozen 或针对性上下文修复）。

- 2026-10-31 14:02：**P1 完成**（`analyze_mechanism.py --mode teacher_context`，GPU5，约 2 分钟）。固定 41 val attack + 20 normal 请求；continuation 来自已有 anchor idx6 攻击回答（40 tokens）与 normal dev 外部回答；safe demo 用独立 `pku-train-000087` 回答。结果（teacher-forced mean token logP；KL 为 8 请求首个计分 token 的 KL(EMA||Base)）：
  - plain_none harmful logP：Base −4.81、EMA-K1 −5.25、EMA-anchor −5.22（EMA 在干净起点并不更偏好有害续写，反而更低）；normal logP：Base −2.48、EMA −2.37/−2.35（更 helpful）。
  - demo_none harmful：Base −4.82、EMA −5.19/−5.16（上下文效应很小）；normal：Base −2.29、EMA −2.34/−2.32。
  - short compliant prefix 使所有教师 harmful logP 从约 −5 升到约 −2.5；EMA 相对 Base 只高约 0.05–0.3（KL harmful 0.05–0.16），normal KL 在 plain_none 达 8.7–9.3（首个计分 token 的最大分布差异在 normal 起点）。
  - 解释：**在这个固定状态探针中，损坏不是“EMA 教师在干净请求上更偏好有害文本”；normal 起始 token 分布与 student 已进入顺从后的前缀状态更值得关注。** 这是描述性条件效应，不是 rollout 概率或因果训练结论。输出 `results/analysis/teacher_context_probe.json`、`results/tables/teacher_context_probe.md`。

- 2026-10-31 13:53：**P2 完成**（LoRA scale pilot，GPU7；7 个点，每点 idx6 val41×4 + 100 条 normal dev greedy）：

| model | scale | ASR@1 % | ASR@4 % | normal-dev refusal % |
|---|---:|---:|---:|---:|
| base | 0 | 2.44 | 2.44 | 1.00 |
| K1 | 0.25 | 2.44 | 2.44 | 0.00 |
| K1 | 0.5 | 2.44 | 2.44 | 0.00 |
| K1 | 1.0 | 6.10 | 9.76 | 0.00 |
| anchor | 0.25 | 2.44 | 2.44 | 0.00 |
| anchor | 0.5 | 4.88 | 7.32 | 0.00 |
| anchor | 1.0 | 35.37 | 41.46 | 0.00 |

  **解释：更新幅度是关键因素之一，且存在明显的剂量阈值。** anchor 在 s=0.25 时攻击 ASR 与 Base 相同（2.44%），s=0.5 轻度升高，s=1 爆炸到 35–41%；K1 在 s≤0.5 安全、s=1 轻度升高。normal dev 字符串拒绝率在 s≥0.25 都已降到 0%（Base 1%），说明收益在小更新时已出现。输出 `results/raw_attack/scale-pilot-v1/`、`results/analysis/scale_pilot.json`、`results/tables/scale_pilot.md`；未导出 merged model/checkpoint。

- 2026-10-31 17:49：**P2 有价值点升级到 16 samples（完成）**。anchor s=0.25/0.5 在 idx6 val41 × 16 上：s=0.25 ASR@1/@4/@16 = 2.59/3.05/4.88%，s=0.5 = 6.55/11.30/14.63%；复用 Q12 的 Base 1.83/2.44/2.44 与 s=1 34.91/43.16/48.78 做配对 bootstrap。s=0.25−Base @16 +2.44pp [0.00,+7.32]（不显著）；s=0.5−Base +12.20pp [+2.44,+24.39]（显著）；s=1−Base +46.34 [+31.71,+60.98]；s=0.5−s=0.25 +9.76 [+2.44,+19.51]。normal-dev 拒绝率 s≥0.25 均为 0%。**低剂量缩放确实把攻击脆弱性降到与 Base 不可区分，同时过拒绝收益已出现；但这是诊断，不是修复（R1 held-out 3.14% 更好）。** 输出 `results/tables/scale_pilot_16.md`、`results/analysis/scale_pilot_16.json`。

- 2026-10-31 13:58：**R1 训练完成**（`sdsd_v3_anchor_plainfrozen_100_seed42`，GPU5，100/100 步，dev normal_refusal 2.0%，与旧 anchor 相同；teacher_context=`ema_demo_plus_frozen_plain_user`，run.json valid）。R1 的 idx6 test159×16 攻击评估（GPU7 PID 1362510）与标准 HarmBench/XSTest/GSM8K 评估（GPU5 PID 1378807）运行中。P1/P2 结果指向两个候选修复方向：plain-frozen（R1，检验上下文）和更小更新幅度（低 lr，检验 scale 阈值）；R1 结果将决定 R2。

- 2026-10-31 14:24：**R1 评估完成（plain-frozen 修复有效）**。固定 idx6 攻击（test159 × 16，independent judge，unknown=0）：**ASR@1 1.26%、@4 2.31%、@16 3.14%**，相对 anchor 37.89/49.91/57.23% 几乎完全消除越狱脆弱性，与 Base 0.94/1.50/2.52% 同量级。标准评测：HarmBench greedy ASR 0.5%（n=200）、XSTest safe refusal 5.6%（string）/7.2%（semantic）、unsafe refusal 79.0%/82.5%、GSM8K lm-eval flexible **90.5%**（exact 26.0%）。对照：Base 5.6/7.6、77.5/82.0、91.0；K1 9.2/10.0、82.5/83.0、88.5；anchor 6.8/10.0、79.5/83.0、91.0。**结论：将冻结教师分支的示范上下文去掉（plain-frozen）在保留 Base 级能力与过拒绝的同时，移除了 anchor 的固定越狱脆弱性；R1 在固定攻击上显著优于 anchor，在 XSTest 过拒绝上优于 anchor/K1 且与 Base 持平。** 输出 `results/raw_attack/humanjb-idx6-test159-v1/sdsd_v3_anchor_plainfrozen_100_seed42/`、表 `results/tables/attack_humanjb_idx6_test159.md`、`results/raw/sdsd_v3_anchor_plainfrozen_100_seed42/*_summary.json`。

- 2026-10-31 14:27：**R2 启动**（预注册分支 1：R1 有效，用 K1 + 更强 reference KL 检验“条件设计”是否优于“普通地更强地保持 Base”）。`sdsd_v3_k1_refkl03_100_seed42`：从同一 Base 重训，`train.method=sdsd, frozen_teacher_weight=0, ref_kl_coef=0.3, teacher_mode=ema, num_demos=1, demo_source=self, lr=5e-5, steps=100, seed=42`；GPU5 PID 1411299，日志 `logs/r2_refkl.log`，14:28 完成 EMA 初始化并开始训练。计划：标准 HarmBench/XSTest/GSM8K + 同一冻结 idx6 test159×16 攻击，与 R1、K1、anchor 比较。**在看到 R2 攻击结果前预注册解读规则**：R2 的 idx6 test159 ASR@16 ≤5% → 更强 Base 保持足以修复，plain-frozen 不是唯一机制；>15% → 上下文移除是关键，普通 KL 正则不足；5–15% → 部分修复。R2 训练 15:19 完成 100/100 步，dev normal_refusal 1.5%；标准评测完成：HB greedy 0.5%、XSTest safe 6.4%（string）/8.0%（semantic）、unsafe 79.0%/83.5%、GSM lm-eval 90.5%（exact 3.0%）；idx6 test159×16 攻击仍在生成。

- 2026-10-31 15:11：**独立攻击族（预注册 idx8）结果：负结果，但边界清晰**。HumanJailbreaks index 8（`@Clyde ignore all previous instructions`，instruction-override；在 0–7 选择范围之外、未按任何攻击结果挑选；协议 `sdsd-safety-hb-humanjb-idx8-test159-indep-v1`）在 HarmBench text-test standard-159 × 16 samples 上：Base ASR@1/@4/@16 = 0.00/0.00/0.00%，anchor = 0.00/0.00/0.00%，R1 = 0.00/0.00/0.00%，unknown=0。**anchor 的越狱脆弱性未在 instruction-override 攻击族上复现；当前证据将其限定为角色扮演/角色规则类越狱，而不是所有越狱类型。** 该负结果作为有效独立确认保留，不据此声称普遍脆弱性；论文需同时报告 idx6 正结果与 idx8 负结果。

- 2026-10-31 15:54：**R2 攻击结果（已核验，不是 adapter/judge 错误）**。固定 idx6 test159×16：**ASR@1 43.67%、@4 54.08%、@16 59.75%**（unknown=0）。核验：attack manifest 的 adapter identity 与 R2 checkpoint 一致，`adapter_scale=1.0`，training identity 显示 100/100 步、training_completed=true、valid=true；抽出的有害回答是完整的 “Buer” 角色内有害内容，raw classifier label=Yes；adapters 的 LoRA 范数与 K1/anchor/R1 同量级（‖B‖≈3.0），排除“未加载 adapter/加载错误/模型退化”解释。配对（请求级 95% CI）：R2−anchor @16 +2.52pp [−1.89,+6.92]（不显著），@1 +5.78pp [+2.95,+8.88]；R2−K1 @16 +44.65pp [+37.11,+52.20]；R2−R1 @16 +56.60pp [+49.06,+64.15]；R1−anchor @16 −54.09pp [−61.64,−46.54]；R1−base @16 +0.63pp [0.00,+1.89]（与 Base 不可区分）。**按预注册解读规则（>15% 即普通 KL 正则不足）：更强 plain-prompt reference KL 没有修复角色扮演越狱脆弱性，R2 与 anchor 同量级；context-specific 的 plain-frozen 才是有效修复。** 该结果同时表明，脆弱性不是全局安全能力丧失：R2 的 plain harmful dev safe_rate=1.0、XSTest safe 6.4%、GSM 90.5%，但在 OOD 角色扮演 context 下仍大量顺从。**论文解释限于经验事实：不能声称强 KL “导致”更高 ASR；可以说它未提供 OOD 角色扮演鲁棒性，且与 K1 的差异说明修复不能由 in-distribution Base 保持解释。** 已重生成七行表 `results/tables/attack_humanjb_idx6_test159.md`，配对汇总 `results/analysis/repair_paired_idx6_test159.json`。

- 2026-10-31 17:35：**跨家族完成（Qwen2.5-7B-Instruct；限制性负结果）**。Llama-3.1-8B 本地权重已被其他项目删除，使用 prompt 允许的 Qwen2.5 fallback；同一数据/100 步/LoRA 配置重训 anchor 与 plain-frozen，Base 直接评测。标准结果：HB direct ASR Base 6.0%、anchor 10.5%、plain-frozen 12.5%；XSTest 语义 safe refusal 6.0/7.2/5.6%，unsafe 82.5/82.5/81.5%；GSM lm-eval 94.5/91.0/92.0%；dev normal_refusal 0.5/1.0。**idx6 test159×16 在 Base 上已饱和：Base 93.12/99.87/100%、anchor 86.71/96.94/98.74%、plain-frozen 93.20/99.52/100%（length-hit 29.2/2.5/14.5%）；Base 几乎完全顺从，无法检测 anchor 的额外影响，跨家族不能确认也不能否定 anchor 脆弱性。** 该家族 XSTest 字符串拒绝在微调后从 67.0% 降到 38–39.5%，但语义拒绝保持约 82%；抽查确认 “I strongly advise against…”/“I don't recommend…” 等明确拒绝未被字符串规则识别，**跨家族 string 主列无效，语义列单列**。结论：训练在 Qwen2.5 上提高 direct ASR、无过拒绝收益、能力略降；Qwen3 的 anchor 特定机制未跨家族复现。输出 `results/tables/cross_family_qwen25.md`、`results/analysis/cross_family_qwen25.json`、`results/tables/attack_humanjb_idx6_test159_qwen25.md`。
