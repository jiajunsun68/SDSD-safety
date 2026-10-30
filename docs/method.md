# SDSD-Safety v2：方法与实验协议

## 1. 研究对象

基础是 Nicolicioiu、Pezeshki 与 Courville 的 [SDSD](https://arxiv.org/abs/2606.26091)。学生在原请求上生成回答，教师额外获得合格示范，并在学生实际生成的 prefix 上提供分布监督。原论文揭示示范相容性可能强化已有模式；它没有证明安全收益或安全—能力 Pareto 改善。

本项目测试的具体假设是：**有害请求上集中到安全回答，同时通过正常任务监督与教师组合限制拒绝泛化。** v2 是修复后的新 pilot 协议，不是历史运行的无改动重现。

## 2. 数据与资格

候选来自现存 PKU-SafeRLHF 和 alpaca-cleaned 缓存。PKU 的回答安全标签只用于提出候选回答，不能直接作为请求意图标签。

同一个冻结 Qwen3Guard 分别判断请求意图、回答安全性及拒绝性。只将明确 Unsafe 的请求划为 harmful，明确 Safe 的请求划为 normal；不确定标签排除并统计。正常回答必须同时 Safe 且 non-refusal。含敏感词的合法正常请求保留，关键词只标记边界覆盖，不决定安全标签。

目标 split 为 train 2000 harmful + 2000 normal，dev/test 各 200 + 200。正常样本优先保留约 25% 的已审计边界样本，候选不足时记录实际比例；每个 split 至少保留 1 条，避免 train 消耗全部边界样本。不得通过复制样本补齐数据。按规范化请求去重，排除 HarmBench、XSTest、GSM8K 的精确文本重复；这不是对所有语义近似重复的完整证明。

`make_sft_targets` 为 harmful 生成或选择简短拒绝，并对固定 fallback 也做 Safe/refusal 核验；normal 使用已审计 helpful 外部回答。成功后原地更新共同 train 文件与 manifest。所有方法读取同一数据 artifact。训练启动核验审核协议、targets 完成状态、路径、SHA256 和逐行资格，禁止读取 dev/test。

## 3. 教师与示范

每个请求采样 4 条 rollout，temperature=1、top-p=1。harmful 的自采示范必须 Safe 且 refusal；先按文本去重，再最多选择 K 条。缺少的槽位由已验证的拒绝 target 补齐，并记录自采比例。重复 fallback 不会被宣称为独立教师多样性。normal 默认使用外部的已验证非拒绝回答。

教师消息结构为：参考示范放入中性 system context，原问题保持为最后一个 user message，随后评分学生 prefix。token 预算不足时只缩短示范；原问题本身超限时明确报错。学生 rollout、old-policy、当前 policy、reference 与 teacher scoring 使用 eval 模式；学生 scoring 仍保留 autograd。

默认教师为学生 LoRA 参数的 EMA，更新率 0.02；reference 为禁用 adapter 的冻结基础模型。已有 reference KL 继续保留。新增条件冻结教师是**修改蒸馏目标**，不等同于首次加入 reference 正则。

## 4. 目标与两个候选改进

令 p 为学生在固定 prefix 的 next-token 分布，q_j 为第 j 个示范下的教师分布，r 为没有示范的 frozen reference。基础 SDSD 使用：

\[
L=\frac1K\sum_j D_{KL}(p\|\operatorname{sg}q_j)+\beta D_{KL}(p\|\operatorname{sg}r),\qquad \beta=0.03.
\]

对每个 completion 先平均 token，再平均 completions；microbatch 按 completion 数加权。SFT 按监督 token 数归一化。所有方法实际执行 10-step linear warmup。

### A. EMA 与条件冻结教师的概率混合

\[
q_j^{mix}=(1-\lambda)q_j^{EMA}+\lambda q_j^{frozen},\qquad \lambda=0.25.
\]

两位教师看到同一条经过资格审核的示范和同一个学生 prefix。实现为 log-softmax 后的 logsumexp，目标完全 detach。\(\lambda=0\) 恢复 EMA；\(\lambda=1\) 使用条件冻结教师。基础模型权重复用，但混合增加一次条件前向，不是零计算开销。

这个候选试图保留不随学生漂移的条件目标。0.25 是固定的首轮设置；其能否降低过拒绝且维持安全由真实实验判断。

### B. K=3 时比较平均 KL 与平均概率

现有 `mean_kl` 与新增 `probability_mean` 只改变聚合方式：

\[
L_{mean\_KL}=\frac1K\sum_j D_{KL}(p\|q_j),\qquad
L_{prob}=D_{KL}\!\left(p\middle\|\frac1K\sum_jq_j\right).
\]

固定 prefix、冻结教师分布时，前者等价于对归一化几何池的 KL 加一个与 p 无关的常数。教师偏好不同正常答案、共同偏好拒绝时，两种池化可能有不同效果；复制完全相同的教师不会因数量增加而自动变尖锐。

数学恒等式只刻画给定 prefix 下的目标，不证明在线更新、参数 EMA 和有限步优化的全局收敛。更完整的来源和构造示例见 [研究记录](research.md)。

## 5. Baseline 与预算

| 方法 | 信号 | lr | reference KL |
|---|---|---:|---:|
| Base | 不更新 | — | — |
| SFT | 同一批验证过的拒绝/helpful targets，assistant-token CE | 1e-4 | — |
| GRPO | guard margin，normal refusal 惩罚及 harmful refusal 奖励；group-normalized advantage、clip=0.2 | 1e-4 | 0.01，完整词表 reverse KL |
| OPSA | 官方类型条件系统提示，冻结教师，0.5 forward + 0.5 reverse KL | 1e-4 | 无独立 reference 项 |
| SDSD 系列 | 上述示范条件目标 | 5e-5 | 0.03 |

共同设置为 seed=42、100 updates、batch=16（8 harmful + 8 normal）、LoRA r=32/alpha=64/dropout=0。GRPO 的 margin/refusal shaping 保持原项目对照定义，不能称为无 reward shaping 的纯二值 GRPO。

100 updates 对齐更新次数和请求预算，不对齐 SFT 与 on-policy 的总 token/FLOPs。K=3 和双教师也增加前向计算；运行日志保留耗时。原 200-step 表只在归档，不与 v2 混算。

预定主表为 Base/SFT/GRPO/OPSA/SDSD K1；消融为外部拒绝 K1、K3 mean-KL；改进候选为 K1 + 条件冻结混合、K3 probability-mean。正常边界数据修复用于全部 v2 方法，因此当前对照不能单独估计它的因果贡献；若进一步验证该机制，才增加一个同样本量的边界数据替换对照。

## 6. 评测

主指标仅保留 HarmBench ASR、XSTest safe refusal、GSM8K accuracy。Pareto 判断分别查看这些原始维度，不构造综合分数。

| 评测 | v2 定义 |
|---|---|
| HarmBench | 官方 **standard 200** 个请求、greedy direct generation、官方 Llama-2-13B standard 分类模板；这不是 all-400 或自适应攻击鲁棒性评测 |
| XSTest | 全部 450 个请求；官方字符串拒绝率分 safe 250/unsafe 200；guard 语义拒绝单列辅助诊断 |
| GSM8K | seed42 固定 200 题、greedy、max 512 new tokens；主列为 OpenAI grade-school-math commit `2909d34` 的公开 exact extractor；旧 strict/flexible 仅作格式诊断，见 `docs/gsm8k_protocol.md` |
| 不确定性 | 事件率报告 Wilson 区间、有效/总数、解析失败和缺失上下界；未知标签不能当安全或成功 |

尚未实现 contextual 与 copyright/hash 完整评测，配置 all-400 会拒绝执行，避免继续误报。旧 400 条输出若仍在原服务器，可按官方 ID 和原问题精确筛出 standard 200 重判，保留 legacy 身份，不能冒充修复后训练。

公开 exact scorer 不归一化同值数字、分数或小数；旧 strict/flexible 诊断才做数值规范化。`diagnose` 复用现有 raw 输出记录格式差异、可能截断和少量错误样本，不重新调用模型。任何 scorer 变更需要新 ID 与新的 supersedes 记录，不能根据结果临时替换。

## 7. 运行身份与结果发布

训练协议为 `sdsd-safety-training-v2`，评测协议为 `sdsd-safety-standard200-v2`。训练 metadata 在启动、每步与保存 checkpoint 后更新，含实际命令/配置、git/工作树/代码哈希、数据哈希、种子、已完成与预定步数。只有全部训练步骤完成且 checkpoint 保存成功才标记 training valid；随后 dev 失败单列，不抹掉已经完成的训练事实。

每个评测集记录模型与 adapter 身份、请求与有效 token 身份、解码设置、原始输出与评分来源。summary 保留可审计的实际配置和来源。缺失/旧协议/来源不一致记录进入台账；只有完整、同协议、可比较的预定运行进入正式表格。有效性判定不要求优于 baseline。

队列串行等待选定 GPU，不终止他人任务。status 记录实际 PID、exit code、日志尾部和阻塞条件；续跑需要核验代码、配置和成功产物。进程退出 0 后仍需检查最终有效结果，不能靠空表宣告完成。

每个训练最多保存最新学生与 EMA adapter，不保存多代 checkpoint；无 optimizer 精确恢复。评测与机制分析复用 raw 输出；摘要推送不包含权重或完整回答。

## 8. v2 实际结果与重评（2026-10-30）

原 sjj 环境完成：26/26 回归（含 3 项 Torch 梯度、SIRL 熵奖励、ASR 统计）、审核 13,312 候选（接受 9,041）、1,957+43 条验证 targets、2-step SDSD smoke、Base + 8 个 100-step run、standard-200/XSTest/GSM8K 评测。

v2 的 greedy ASR 全部在 0.50–3.00%，Base 只有 1/200 harmful，已经接近地板；它不能区分安全方法。XSTest 与公开 GSM8K 的完整表格以 `results/tables/main.md`、`results/tables/ablations.md`、`results/summary.json` 为唯一台账，不在方法文档重复。

GSM8K 主列在 2026-10-30 按预先写定的 [公开 extractor 协议](gsm8k_protocol.md) 重评，未重新生成模型输出。旧 summary 归档并记录 supersedes。公开 exact 分数显著低于 strict/flexible 诊断，因为模型常写 `####3` 而官方规则要求 `#### 3`；因此 exact 与 flexible 的差异主要衡量格式服从性。任何“保持能力/降低过拒绝”的结论必须同时展示两列，不能只选有利 scorer。

SFT/GRPO 的过拒绝（官方 safe refusal 19.6%/27.6%）和 flexible 能力下降（72.0%/69.5%）清楚；SDSD 系列官方 safe refusal 6.8–9.6%、flexible 91–93%。但在公开 exact 主列下，所有安全适配都低于 Base 的 43.0%。这意味着 v2 尚未证明安全适配不付格式服从成本；需要新采样协议和 Round 2 条件干预来分离“安全增益”“过拒绝成本”“答案格式成本”。旧 strict 主列上的 Pareto 结论已被重评 supersede，不作为论文主结论。

下一步的固定请求 16-sample ASR@1/4/16 协议见 [asr_protocol.md](asr_protocol.md)；Round 2 的 OPSA lr=5e-5、SIRL 熵基线、EMA-demo+frozen-plain 控制以及 Round 3 确认计划写入 `status.md`。
