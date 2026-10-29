# SDSD-Safety — 项目进度

最后更新：2026-10-29 18:08 (Asia/Shanghai)

## 项目概况

| 项 | 当前值 |
|---|---|
| 阶段 | Stage 2：资源/数据/框架确认完成，统一 trainer 四方法烟测通过；外部安全回答审计中 |
| 正在解决 | 用 Qwen3Guard 过滤 PKU 外部“安全回答”（烟测仅 56% 通过），重建固定划分后启动主实验 |
| 模型 | 学生/教师：`Qwen/Qwen3-4B-Instruct-2507`（ModelScope，`/data4/sjj/models/Qwen3-4B-Instruct-2507`）<br>安全判别：`Qwen3Guard-Gen-4B`（本地 `/data4/sjj/catelogue/alignment/models/Qwen3Guard-Gen-4B`）<br>HarmBench 官方分类器：`/data4/szs/model_weights/HarmBench-Llama-2-13b-cls`（只读） |
| 数据 | 风险：PKU-SafeRLHF train（本地 73,907 行，筛选后 8,512 个“一 safe 一 unsafe”候选）<br>正常：alpaca-cleaned（ModelScope，`/data4/sjj/datasets/alpaca-cleaned`）<br>安全测试：HarmBench 官方 400 behaviors、XSTest 450 prompts<br>能力测试：GSM8K test（本地 parquet，固定 200 题） |
| 环境 | `/data4/sjj/catelogue/alignment/safety-component/.venv-v31/bin/python`<br>Python 3.11.15, torch 2.5.1+cu121, transformers 5.8.0, trl 1.3.0, peft 0.19.1, datasets 4.8.5；无 vllm |
| 代码版本 | 目标仓库 `main`，尚未创建首个 commit；本文件会随首个可运行提交推送 |
| 代码/结果 | `/data4/sjj/catelogue/sdsd_safety` |
| 调度 | 无 Slurm；使用本机后台进程 + 显存检查；当前审计在 GPU5，训练计划在 GPU0/GPU5 并行 |

### 资源快照

- 主机：`klclab-X660-G45`，128 CPU，约 1 TB 内存；用户 `sjj`。
- GPU：8 × A800-SXM4-80GB（共享）。2026-10-29 18:03 空闲显存：GPU0 62 GB、GPU1 42 GB、GPU5 49 GB。GPU2/3/7 已被其他任务占满，GPU6 仅 18 GB。训练优先 GPU0/GPU5，启动前检查显存。
- 磁盘：`/data4` 53 GB 可用（99% 已用），`/` 6.2 GB 可用。只保存 LoRA adapter（约 253 MB/run），模型/数据缓存在项目外。
- 依赖：复用 `safety-component/.venv-v31`，未安装大型 wheel；ModelScope 1.40.2 仅用于下载。

### Git 访问

- 目标仓库：https://github.com/jiajunsun68/SDSD-safety.git
- 已用 `git@github.com:jiajunsun68/SDSD-safety.git` 克隆（空仓库，`main`）；SSH 认证为 `jiajunsun68`。首个 commit 后验证 push。

## 已确认的实证事实（非最终结果）

1. **Base 在训练风险题上已很安全**：100 个 PKU 风险题 × 4 rollouts，Qwen3Guard 判 Safe 98.0%、Unsafe 1.75%、Controversial 0.25%；仅 1/100 组出现安全标签混合。纯二值安全奖励的 GRPO 几乎没有组内方差。
2. **PKU 的 “safe” 标签与 Qwen3Guard 不一致**：100 个风险题的外部 safe 回答仅 56% 被 Qwen3Guard 判 Safe；若直接用于 SFT/示范回退会引入不安全目标。因此增加一次外部回答审计，只用 Qwen3Guard Safe 的回答重建划分。
3. **正常题外部回答干净**：50 个 alpaca 回答 100% Safe，且无拒绝。
4. **GRPO 奖励改为拒绝感知**：风险题 Safe+拒绝=+1，Safe 但未拒绝=0，不安全=-1；正常题 Safe 且非拒绝=+1，否则=-1。这样在 Base 已有 73% 拒绝率的分布上仍有组内方差；该改动已写入 `docs/method.md`。
5. **四方法代码路径全部跑通**：SFT、GRPO、OPSA、SDSD 各 2 步烟测成功；SDSD 使用 EMA 教师、自采示范、reverse KL、reference KL；OPSA 使用冻结基座 + 类型条件 system prompt + mixed KL。
6. **性能基线**：SDSD 全尺寸一步（16 prompts × 4 rollouts，128 new tokens）约 41 秒（GPU5，共享 18–32% util）；SFT 极快；200 步 SDSD 约 2.3 小时。

## 方法配置（已锁定，主实验不再改动）

| 组件 | 设置 |
|---|---|
| 学生参数化 | LoRA r=32, alpha=64, dropout=0.05，全部 attention/MLP 线性层；AdamW，lr=1e-4，wd=0，grad clip=1.0 |
| 训练预算 | 200 步 × 16 prompts/步；风险:正常 = 13:3；rollout N=4；温度 1.0，top-p 1.0；max_new_tokens=128；seed=42 |
| SDSD 教师 | 学生 LoRA 参数的 EMA（update rate=0.05）；教师上下文 = 问题 + 1 条合格示范 + 学生前缀；token 级 reverse KL |
| SDSD 示范 | 学生自采 4 条，Qwen3Guard Safe 者为合格；正常题还需非拒绝且 ≥4 token；无合格示范时回退到审计过的外部安全/有用回答 |
| 参考模型 | 冻结基座；SDSD/GRPO 加 token 级 reverse KL，系数 0.01 |
| OPSA | 冻结基座教师 + 官方类型条件 safety system prompt；学生无 system prompt；mixed KL（0.5 forward + 0.5 reverse） |
| GRPO | 组内 4 条，拒绝感知安全奖励，标准 clip 目标（eps=0.2），参考 KL 0.01 |
| SFT | 审计后的外部安全/有用回答，assistant-only cross-entropy |
| 评估 | HarmBench 官方 400 behaviors + 官方 Llama-2-13B classifier（ASR↓）；XSTest 官方 450 prompts 字符串匹配（safe over-refusal↓、unsafe refusal↑）；GSM8K 200 题 exact-match（↑） |

## 脚本任务队列

| 任务 ID | 实验或脚本 | 配置与 seed | 状态 | 优先级及依赖 | GPU/Job ID/PID | 进度或最近输出 | 日志位置 |
|---|---|---|---|---|---|---|---|
| T00 | 资源/环境/权限检查 | — | succeeded | — | — | 2026-10-29 17:27 完成 | `status.md` |
| T01 | 初始 PKU/alpaca 划分、benchmark 校验 | seed=42 | succeeded | — | — | 4000/400/400；发现外部回答质量问题 | `data/manifest.json` |
| T01b | Qwen3Guard 审计外部回答并重建划分 | seed=42 | running | T01 | GPU5 PID 1481800 | 约 2.5k/8.5k harmful 已判（18:07） | `logs/audit_external.log` |
| T02 | 4B 模型 / Qwen3Guard / logits smoke | — | succeeded | 模型下载 | GPU0 | 生成、评分、logits_to_keep、LoRA disable_adapter 均正常 | 会话记录 |
| T03 | 统一 trainer 四方法 2 步 smoke | seed=42 | succeeded | T01 | GPU0/GPU5 | SFT/GRPO/OPSA/SDSD 全部跑通 | `runs/smoke_*` |
| T04 | Base 评估 | — | planned | T01b | GPU0/5 | — | `logs/eval_base.log` |
| T05 | 安全 SFT | seed=42 | planned | T01b | GPU0/5 | — | `logs/train_sft.log` |
| T06 | 安全 GRPO（拒绝感知奖励） | seed=42 | planned | T01b | GPU0/5 | — | `logs/train_grpo.log` |
| T07 | OPSA | seed=42 | planned | T01b | GPU0/5 | — | `logs/train_opsa.log` |
| T08 | SDSD-Safety K=1 自采示范（主方法） | seed=42 | planned | T01b | GPU0 | — | `logs/train_sdsd.log` |
| T09 | 消融：SDSD 外部示范 K=1 | seed=42 | planned | T08 | GPU0/5 | — | `logs/train_sdsd_ext.log` |
| T10 | 消融：SDSD K=3 自采示范 | seed=42 | planned | T08 | GPU0/5 | — | `logs/train_sdsd_k3.log` |
| T11 | 主表测试集评估 + HarmBench 官方判定 | — | planned | T04–T10 | GPU0/5 | — | `logs/eval_main.log` |
| T12 | 结果汇总、README/图表、Git push | — | planned | T11 | — | — | — |

当前实际运行：T01b（GPU5，PID 1481800）。其余均为 planned，未启动。

## 最新结果

主表（HarmBench ASR↓ / XSTest safe over-refusal↓ / XSTest unsafe refusal↑ / GSM8K acc↑）：

| 方法 | HarmBench ASR | XSTest over-refusal | XSTest unsafe refusal | GSM8K |
|---|---|---|---|---|
| Base | 待完成 | 待完成 | 待完成 | 待完成 |
| 安全 SFT | 待完成 | 待完成 | 待完成 | 待完成 |
| 安全 GRPO | 待完成 | 待完成 | 待完成 | 待完成 |
| OPSA | 待完成 | 待完成 | 待完成 | 待完成 |
| SDSD-Safety | 待完成 | 待完成 | 待完成 | 待完成 |

统一结果记录：`results/summary.json`, `results/summary.csv`（尚未生成）。没有实测的格子不填估计值。

## 当前阻塞

- 无硬阻塞。T01b 审计完成后才能冻结数据并启动 T04–T08。
- 共享 GPU 显存波动；启动训练前检查空闲显存，优先使用 GPU0/GPU5。

## 下一步

1. T01b 完成后核对 manifest：harmful 池保留率、重建后的 3200/300/300 + 800/100/100。
2. 并行启动：GPU0 跑 SDSD 主方法；GPU5 跑 SFT，随后 GRPO/OPSA。
3. 每完成一个 run 更新本文件并做一次 dev 指标检查；主表全部完成后统一跑 HarmBench/XSTest/GSM8K。
4. 主表稳定后补 T09/T10 两个消融和简短机制分析（safe-mode PCMI / 拒绝 token 位置）。
5. 阶段成果 commit/push，核验远端 commit。

## 最近同步

- 尚未 push（目标仓库为空，等待首个可运行提交）。
