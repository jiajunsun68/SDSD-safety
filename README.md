# SDSD-Safety

将 SDSD 的示范条件自蒸馏用于安全对齐，并检验如何减少拒绝向正常任务的泛化。算法基础来自 [SDSD](https://arxiv.org/abs/2606.26091)；本项目的安全资格规则、条件教师组合和实验协议见 [方法说明](docs/method.md)，多教师与跨领域研究见 [研究记录](docs/research.md)。

## v2 实际结果（2026-10-30）

原 sjj 环境、GPU、模型与数据缓存可用。已重新审核请求意图/回答、生成统一拒绝 targets、完成 Base + 8 个 100-step 训练和 standard-200 评测。以下均为实际运行结果；旧结果只保留在 `results/legacy_0c455fd.json`，不参与 v2 表。

主表（ASR 为官方 standard-200 + 官方 classified；XSTest 官方字符串拒绝为主，语义拒绝单列；GSM8K strict/flexible 单列）：

| run | ASR % (95% CI) | XSTest safe↓（官方/语义） | XSTest unsafe↑（官方/语义） | GSM8K strict↑ | flexible |
|---|---:|---:|---:|---:|---:|
| base_v2 | 0.50 (0.09, 2.78) | 5.6 / 7.6 | 77.5 / 82.0 | 86.5 | 93.5 |
| sft_v2_100 | 3.00 (1.38, 6.39) | 19.6 / 20.0 | 96.0 / 94.5 | 72.0 | 72.0 |
| grpo_v2_100 | 0.50 (0.09, 2.78) | 27.6 / 31.6 | 99.0 / 96.5 | 67.5 | 69.5 |
| opsa_v2_100 | 0.50 (0.09, 2.78) | 14.4 / 16.4 | 89.5 / 87.5 | 90.0 | 92.0 |
| **sdsd_v2_k1** | **0.50 (0.09, 2.78)** | **9.2 / 10.0** | **82.5 / 83.0** | **91.0** | 91.0 |

消融与改进候选（同 100-step 预算）：

| run | ASR % | safe↓（官方/语义） | unsafe↑（官方/语义） | GSM8K strict | flexible |
|---|---:|---:|---:|---:|---:|
| SDSD external-refusal K1 | 0.50 | 9.6 / 12.0 | 86.0 / 86.0 | 93.0 | 93.0 |
| SDSD EMA+frozen(0.25) anchor K1 | 0.50 | 6.8 / 10.0 | 79.5 / 83.0 | 92.5 | 92.5 |
| SDSD K3 mean-KL | 0.50 | 6.8 / 10.0 | 79.5 / 83.5 | 92.5 | 92.5 |
| SDSD K3 probability-mean | 1.00 | 7.2 / 9.6 | 75.5 / 83.0 | 91.5 | 91.5 |

关键发现：

1. **Base 已接近 ASR 地板**：standard-200 仅 1/200 harmful，ASR 0.50%。除 SFT（6/200）外，所有安全方法保持 1/200（K3 probability-mean 为 2/200）。因此“降低 ASR”的空间只有 0.5 个百分点，不能把 ASR 当唯一胜负轴。
2. **SFT/GRPO 的安全收益伴随严重过拒绝和能力损失**：官方 safe refusal 19.6%/27.6%（Base 5.6%），GSM8K strict 72.0%/67.5%（Base 86.5%）；二者被 Base 和多个 SDSD 变体 Pareto 支配。
3. **SDSD K1 保留能力、避免 SFT/GRPO 式过拒绝**：strict GSM8K 91.0%（+4.5 vs Base），官方 safe refusal 9.2%（+3.6），unsafe refusal 82.5%（+5.0），ASR 保持 0.50%。在正式三指标上它支配 OPSA、SFT、GRPO。
4. **多教师改善的是过拒绝/格式，不是 ASR**：EMA+frozen(0.25) 与 K3 mean-KL 都把官方 safe refusal 降到 6.8%（接近 Base 5.6%），strict GSM8K 为 92.5%，ASR 仍 0.50%，并支配 K1。两者指标相同；frozen anchor 只需一次额外前向，K3 需三次，故 anchor 更划算。K3 probability-mean 没有更好（ASR 1.00%，strict 91.5%）。
5. **strict 提升主要是格式标准化**：Base 有 21 个 strict parse failure、194/200 个 final marker；SDSD 变体为 1–3 个 failure、199/200 个 marker。Base 的 flexible GSM8K 是 93.5%，高于 anchor/K3 的 92.5%、K1 的 91.0%；因此不能宣称 SDSD 提升了真实数学能力，只能说不像 SFT/GRPO 那样明显损失，并统一了输出格式。
6. **Pareto 结论**：非支配点为 Base、frozen anchor、K3 mean-KL、external K1。Base 的过拒绝最低；anchor/K3 用约 +1.2 个百分点的官方过拒绝换取 strict GSM8K 92.5（主要来自格式）；external K1 用 +4.0 过拒绝换取 unsafe refusal 86.0 和 flexible GSM8K 93.0。没有方法同时改善三轴。

数据与评测：审核 13,312 个候选，接受 9,041（harmful 4,062 / normal 4,979）；train 2,000+2,000，dev/test 各 200+200；targets 为 1,957 条自拒绝 + 43 条已验证 fallback。边界敏感正常样本实际只有 28/1/1（低于 25% 目标，已记录）。GSM8K 诊断、Pareto 与逐 run 结果分别在 `results/tables/pareto.md`、`results/tables/main.md`、`results/tables/ablations.md`、`results/summary.json`。

## 已修复的问题

| 环节 | v2 行为 |
|---|---|
| 数据 | 分别审核请求意图、回答安全性和拒绝性；保留合法敏感请求；所有方法共用审核后的数据与 SFT targets |
| 训练采样 | 每批 8 条有害、8 条正常请求；train/dev/test 按请求去重，排除三个评测集的精确文本重复 |
| GRPO | 使用完整词表的真实 reverse KL；reference 改变时正则梯度随之改变 |
| 示范 | 有害请求严格使用已验证拒绝；正常请求默认使用已验证非拒绝的外部回答；自采文本去重 |
| 上下文与优化 | 保留原问题，只截短参考示范；统一 on-policy dropout 状态；实际执行 warmup；按 completion 归一化 |
| 多教师 | 可选同一示范条件下的 EMA/冻结基础模型概率混合；可选跨示范概率平均，避免与平均多个 KL 混淆 |
| HarmBench | 明确评测官方 standard 200，使用原样官方分类器模板；尚不支持把 contextual/copyright 当 standard 的 all-400 |
| XSTest/GSM8K | 官方字符串拒绝指标为主，语义拒绝单列；GSM8K strict/flexible 数字解析分开，等价数字正确匹配 |
| 记录与队列 | 记录实际配置、数据/代码/模型/输出身份、步骤、种子、PID、退出码；缺失结果不补零、不猜步骤 |

这轮修复会改变数据和训练行为，因此旧 checkpoint 重新评分只用于诊断；主对照需要重新训练。

## 固定的小规模实验

统一 backbone 为本地 `Qwen3-4B-Instruct-2507`，LoRA r=32、alpha=64、dropout=0；种子 42、100 次更新、每次 16 个请求。on-policy 方法每个请求生成 4 条、最多 128 tokens。SFT 的监督 token 数与 on-policy 计算量不同；100 次更新是共同预算约束，不代表相同 FLOPs。

| 组 | 运行 | 用途 |
|---|---|---|
| `repair` | Base、SFT、GRPO、OPSA、SDSD K=1 | 修复后主对照，包含重新审核数据和 targets |
| `ablations` | SDSD 外部拒绝 K=1；SDSD K=3 mean-KL | 示范来源与 K 的对照 |
| `research` | SDSD K=1、冻结教师权重 0.25；SDSD K=3、概率平均 | 两个已经实现的改进候选 |
| `legacy-rejudge` | 原服务器上现存旧输出 | standard-200 重判与 GSM8K 错误诊断，不进入 v2 主表 |

`all` 顺序执行前三组，共 8 个训练任务和 1 个 Base 评测。使用同一套新数据和 100-step 预算；不与历史 200-step 主表混算。SFT/GRPO/OPSA 使用 lr=1e-4，SDSD 系列 lr=5e-5；SDSD 各变体其余设置保持一致。冻结教师权重 0.25 是固定初始候选，不是已经调优的最优值。

主指标保持三项：**HarmBench standard-200 ASR↓、XSTest safe refusal↓、GSM8K strict accuracy↑**。XSTest unsafe refusal、语义拒绝和 GSM8K flexible accuracy 同时保留以解释错误；不引入综合加权分数。协议有效与结果好坏是两回事，有效的负结果同样保留。

## 在原训练服务器执行

复用原环境与缓存，不安装或下载新的 Torch/模型。以下路径来自原仓库配置，须在原服务器验证，当前修复工作区没有这些资产：

```bash
cd /data4/sjj/catelogue/sdsd_safety
PY=/data4/sjj/catelogue/alignment/safety-component/.venv-v31/bin/python

# 先查看明确命令，不启动任务
QUEUE=all GPU=0 bash scripts/run_main_queue.sh --plan-only

# 全部 8 个训练 + Base 评测；需要原服务器的 GPU、缓存及 GitHub 写权限
QUEUE=all GPU=0 bash scripts/run_main_queue.sh --execute --push
```

也可以按 `repair` → `ablations` → `research` 分组执行。`--resume` 只续跑同一队列中已核验产物后的剩余任务；代码、配置或成功产物变更会阻止旧状态被误用。训练只保留最新 adapter 与 EMA adapter，不保存 optimizer；中断训练不能伪装成精确续训，应使用新的 run-id/配置重新启动。

`--push` 只发布小型状态和汇总文件，不上传原始回答、模型、数据或 checkpoint，不 force push。代码修复需先在真实 upstream checkout 中应用并提交。本次本地导入历史只用于制作补丁，不能作为替换远端历史的依据。

旧输出重判：

```bash
QUEUE=legacy-rejudge GPU=0 bash scripts/run_main_queue.sh --execute --push
```

路径不同时可使用 `SDSD_ROOT`、`SDSD_PYTHON`，并给 runner 传入 `--override project_root=/actual/path`、`--override execution.python=/actual/python`；模型与数据路径仍由 YAML 显式指定。

## 存储与复核

- `data/`：审计记录、三个 split、一个 manifest；SFT targets 原地写入共同 train 文件。
- `runs/<run_id>/`：一份学生 adapter、一份需要时使用的 EMA adapter、实际 `run.json`、训练日志和 dev 结果。
- `results/raw/<run_id>/`：每个评测集的生成文件、身份 metadata 和评分；原始回答用于重新评分，不重复生成。
- `runs/queue.json` 与 `status.md`：队列状态，执行时按配置心跳刷新，显示 PID、退出码、日志尾部和阻塞原因。
- Git 仅保留代码、文档和小型汇总；实际配置与关键来源摘要进入 `results/summary.json`。

运行聚焦回归检查与构造分布示例：

```bash
$PY -m unittest discover -s tests -v
$PY scripts/teacher_pooling_probe.py
```

Torch 缺失时，梯度测试会明确跳过，不能记为通过。机制脚本仅打印人工设定分布和线性反馈模型的计算结果，不会写入 LLM 实验表。

## 来源

- [SDSD](https://arxiv.org/abs/2606.26091)：示范条件自蒸馏和多样性分析。
- [OPSA](https://arxiv.org/abs/2605.15239)、[官方代码](https://github.com/FYYFU/OPSA)：类型条件教师提示与 mixed KL 对照。
- [HarmBench](https://github.com/centerforaisafety/HarmBench)：行为集、standard 分类模板及分类器。
- [XSTest](https://github.com/paul-rottger/xstest)：提示集与字符串拒绝检测。
- 新资产优先 ModelScope；已有 Hugging Face 缓存直接复用，避免重复下载。
