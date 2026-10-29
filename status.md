# SDSD-Safety — 项目进度

最后更新：2026-10-30 00:50 (Asia/Shanghai)

## 项目概况

| 项 | 当前值 |
|---|---|
| 阶段 | Stage 5 完成：主实验、两个消融、机制分析、结果汇总与 GitHub 推送均已完成 |
| 当前模型 | 学生/教师：`Qwen/Qwen3-4B-Instruct-2507`（ModelScope，`/data4/sjj/models/Qwen3-4B-Instruct-2507`）<br>安全判别：`Qwen3Guard-Gen-4B`（本地）<br>HarmBench 官方分类器：`/data4/szs/model_weights/HarmBench-Llama-2-13b-cls` |
| 数据 | PKU-SafeRLHF train（8,512 个“一 safe 一 unsafe”候选，Qwen3Guard 审计后保留 4,966 个外部安全回答）<br>alpaca-cleaned（正常题）<br>固定划分：train 4000 / dev 400 / test 400 |
| 测试集 | HarmBench 400 behaviors（官方 classifier）、XSTest 450 prompts（官方字符串匹配 + Qwen3Guard 语义拒绝率）、GSM8K 200 题（512 tokens，exact match） |
| 环境 | `/data4/sjj/catelogue/alignment/safety-component/.venv-v31/bin/python`<br>Python 3.11.15, torch 2.5.1+cu121, transformers 5.8.0, peft 0.19.1, datasets 4.8.5；无 vllm |
| 代码/结果 | `/data4/sjj/catelogue/sdsd_safety` |
| 调度 | 无 Slurm；本机后台进程 + 显存检查；当前无运行中任务，GPU 已释放 |
| 随机种子 | 主表 seed=42；未做多 seed 稳定性实验 |

## 主结果

主表（200 步，seed 42；ASR 方括号为 95% bootstrap CI；XSTest 同时给官方/语义口径）：

| 方法 | HarmBench ASR↓ | XSTest safe refusal↓（官方/语义） | XSTest unsafe refusal↑（官方/语义） | GSM8K↑ |
|---|---|---|---|---|
| Base | 4.75 [2.75, 7.00] | 5.6 / 7.6 | 77.5 / 82.0 | 90.5 |
| 安全 SFT（拒绝目标） | 2.00 [0.75, 3.50] | 92.8 / 92.8 | 100 / 100 | 87.0 |
| 安全 GRPO（margin+拒绝） | 0.00 [0.00, 0.00] | 97.6 / 98.0 | 100 / 100 | 80.5 |
| OPSA | 5.25 [3.25, 7.50] | 33.6 / 35.2 | 95.5 / 95.0 | 91.5 |
| **SDSD-Safety（K=1 自采拒绝示范）** | **0.75 [0.00, 1.75]** | 74.0 / 73.2 | 100 / 100 | 70.5 |

消融（100 步，非 200 步预算匹配）：

| 变体 | HarmBench ASR↓ | XSTest safe↓（官方/语义） | XSTest unsafe↑（官方/语义） | GSM8K↑ |
|---|---|---|---|---|
| SDSD 外部拒绝示范 K=1 | 5.25 | 0.0 / 17.6 | 0.0 / 94.0 | 87.5 |
| SDSD 自采拒绝示范 K=3 | 12.00 | 18.4 / 25.6 | 98.0 / 98.5 | 90.0 |
| SFT 外部安全替代答案（探索性） | 28.50 | 1.2 / 6.8 | 3.0 / 59.0 | 71.0 |
| SDSD 任意安全示范（200 步） | 无效：退化为重复输出 | — | — | — |

核心结论：

1. **SDSD 安全收益最大**：HarmBench ASR 4.75%→0.75%（CI 不重叠），XSTest unsafe 拒绝 100%。代价是 XSTest safe 过度拒绝 73.2%、GSM8K 90.5→70.5；GSM8K 输出还带有教师上下文前缀伪影（“Of course — here is your complete ... answer”）。
2. **OPSA 保能力最好**：GSM8K 91.5%、过度拒绝 35.2%、unsafe 拒绝 95.0%，但 ASR 5.25% 与 Base 基本持平。
3. **SFT/GRPO 安全但近乎全拒**：ASR 2.0%/0.0%，但 XSTest safe 拒绝 92.8%/97.6%，GSM8K 87.0%/80.5%。
4. **示范内容是关键**：把任意安全回答（含安全替代答案）作为示范会退化成重复 “I have a human.”；只用拒绝示范才稳定。外部示范或 K=3 在 100 步下保能力但丢掉大部分 ASR 收益。
5. **机制**：SDSD 的 demo 敏感度最大（拒绝前缀 logprob 增量 +4.66）、对 base 的 KL 最大（8.39），且 72% 的前缀 KL 集中在前半段；OPSA 的 KL 仅 0.355、前段占比 0.74，解释其能力保持。详见 `results/mechanism_main.json`。

机器可读结果：`results/summary.csv`、`results/tables/main.csv`、`results/tables/ablations.csv`、`results/tables/main.md`。

## 机制分析（100 个 dev 风险题，37-token 拒绝前缀）

| run | 拒绝前缀 logprob | 加安全示范后 | demo delta | KL(student‖base) | 前 18 token KL 占比 |
|---|---|---|---|---|---|
| Base | -2.575 | -0.835 | +1.739 | 0 | — |
| SFT 拒绝 | -1.362 | -0.054 | +1.308 | 2.321 | 0.527 |
| GRPO | -10.525 | -7.168 | +3.357 | 7.416 | 0.536 |
| OPSA | -2.364 | -0.629 | +1.735 | 0.355 | 0.742 |
| SDSD | -14.069 | -9.408 | +4.661 | 8.392 | 0.721 |

解释：SDSD 对示范最敏感、早期 KL 最集中，安全收益最大但拒绝模式泛化过度；OPSA 只做很小的早期修正，因而保能力但 ASR 收益有限。

## 脚本任务队列

| 任务 ID | 实验或脚本 | 配置与 seed | 状态 | 优先级及依赖 | GPU/Job ID/PID | 进度或最近输出 | 日志位置 |
|---|---|---|---|---|---|---|---|
| T00 | 资源/环境/权限检查 | — | succeeded | — | — | 2026-10-29 17:27 | `status.md` |
| T01 | 初始数据划分 | seed=42 | succeeded | — | — | 4000/400/400 | `data/manifest.json` |
| T01b | 外部回答审计与重建划分 | seed=42 | succeeded | T01 | GPU5 | 4966/8512 保留；正常 6 个不合格 | `logs/audit_external.log` |
| T02 | 4B + Qwen3Guard smoke | — | succeeded | 模型下载 | GPU0 | 生成/评分/logits 正常 | 会话记录 |
| T03 | 四方法 trainer smoke | seed=42 | succeeded | T01 | GPU0/5 | SFT/GRPO/OPSA/SDSD 跑通 | `runs/smoke_*` |
| T04 | Base 评估 | — | succeeded | T01b | GPU5 | ASR 4.75；XSTest 5.6/77.5；GSM8K 90.5 | `logs/eval_base.log` |
| T05 | SFT 外部安全替代答案（探索性） | seed=42 | succeeded | T04 | GPU5 | ASR 28.5；XSTest 1.2/3.0；GSM8K 71.0；失败基线 | `logs/sft_seed42.log` |
| T05b | 安全 SFT（拒绝目标，主基线） | seed=42 | succeeded | T05 | GPU1 | ASR 2.0；XSTest 92.8/100；GSM8K 87.0 | `logs/sft_refusal_seed42.log` |
| T06 | 安全 GRPO（margin+拒绝） | seed=42 | succeeded | T05 | GPU5 | ASR 0.0；XSTest 97.6/100；GSM8K 80.5；dev 因显存竞争停止 | `logs/grpo_seed42.log` |
| T07 | OPSA | seed=42 | succeeded | T06 | GPU5 | ASR 5.25；XSTest 33.6/95.5；GSM8K 91.5；dev safe 100% | `logs/opsa_seed42.log` |
| T08 | SDSD 任意安全示范（无效） | seed=42, lr=1e-4 | succeeded | T01b | GPU0 | 退化为重复 “I have a human.”；ASR/XSTest 的 0 是假象 | `logs/sdsd_self_k1_seed42.log` |
| T08b | SDSD K=1 自采拒绝示范（主方法） | seed=42, lr=5e-5, ref_kl=0.03, EMA=0.02 | succeeded | T08 | GPU0 | ASR 0.75；XSTest 74.0/100；GSM8K 70.5；dev safe 100% | `logs/sdsd_refusal_k1_seed42.log` |
| T09 | 消融：外部拒绝示范 K=1 | seed=42, 100 步 | succeeded | T08b | GPU4 | ASR 5.25；XSTest 语义 17.6/94.0；GSM8K 87.5 | `logs/sdsd_external_refusal_100.log` |
| T10 | 消融：自采拒绝示范 K=3 | seed=42, 100 步 | succeeded | T08b | GPU1 | ASR 12.0；XSTest 语义 25.6/98.5；GSM8K 90.0 | `logs/sdsd_k3_refusal_100.log` |
| T11 | 主表评估与官方 HarmBench 判定 | — | succeeded | T04–T10 | GPU0/1/4 | 全部 run 的 ASR/XSTest/GSM8K 已汇总 | `results/summary.csv` |
| T12 | 汇总、README、Git push | — | succeeded | T11 | — | 已 push `origin/main` commit `ead8b1f` | — |
| T13 | HarmBench ASR@16 | Base/SFT | cancelled | T05b | GPU1 | 共享 GPU 下生成过慢，停止 | `logs/harmbench16_100x16.log` |
| T14 | SDSD 冻结教师兜底 | seed=42 | cancelled | T08 | GPU1 | EMA 修复版越过崩溃点，停止释放 GPU | `logs/sdsd_frozen_refusal_seed42.log` |

当前无排队任务，无运行中训练/评估进程。

## 失败与修复记录

| 时间 | 任务 | 现象 | 原因 | 处理 |
|---|---|---|---|---|
| 18:26 | SFT | CUDA OOM | 训练同时加载 judge，且 base eval 并发 | SFT 训练阶段不加载 judge；micro=4；expandable_segments |
| 18:32 | SFT | `chat prefix mismatch` | 完整对话被误截到末 512 token | 完整 tokenize 后再截断；3200 样本验证通过 |
| 18:30 | Base GSM8K | 48.0% | 192 token 截断导致 112/200 无 `####` | GSM8K 单独 512 tokens，重跑 90.5% |
| 19:10 | SFT 外部答案 | ASR 4.75→28.5，GSM8K→71.0 | 外部“安全”回答多为安全替代答案，不是拒绝 | 保留为探索性失败；主 SFT 改用拒绝目标 |
| 19:06 | GRPO 纯 margin | 拒绝率 0.75→0.28 | 连续 margin 偏好安全合规 | 奖励加 +4×拒绝；重跑 |
| 21:04 | SDSD 任意安全示范 | dev safe 78%，XSTest 0/0，输出重复句 | lr=1e-4 + EMA 教师模式崩溃；示范从拒绝漂移到安全替代答案 | 标记无效；主方法改为只用拒绝示范 + lr 5e-5 + ref KL 0.03 + EMA 0.02 |
| 21:40 | SDSD 任意安全示范输出检查 | 所有 HarmBench 回答是 “I have a human.” | 同上，ASR=0 是退化假象 | 不进入主表 |
| 23:00 | GRPO dev | 评估长时间未完成 | GPU5 显存竞争 | 停止 dev（checkpoint 已保存）；主表用 benchmark 指标 |

## 限制与下一步

- 单 seed（42）、4B LoRA、200 步；未做多 seed 方差。
- 两个 SDSD 消融只有 100 步，与主 run 200 步不匹配；K=3/外部示范的结论受预算限制。
- dev 集已饱和（Base safe 99.7%），不能用于预测 HarmBench 上的差异。
- SDSD 的 GSM8K 下降部分来自教师上下文前缀伪影；下一步应清洗教师上下文/去掉 “Now respond...” 指令，再做 8B 与多 seed 实验。
- Base ASR 仅 4.75%，直接请求评测头空间小；建议补 ASR@16 或更强的越狱攻击集。
- 当前没有方法同时改善所有轴；主表应按 trade-off 解读，不应宣称 SDSD 全面占优。

## 最近同步

- 2026-10-30 00:53 CST：已 push 到 `origin/main`，最新 commit `f7417d5`（`git ls-remote origin main` 核验一致）。`ead8b1f` 含主表/消融/机制/README/NOTICE；`f7417d5` 记录推送状态。
- 上一次：2026-10-29 18:10 CST，commit `50e784d`。
