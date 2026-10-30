# SDSD-Safety：过拒绝、能力保留与多教师的最小研究方案

日期：2026-10-30。本文最初是修复实验后的候选设计与文献依据；第 7 节现已补入在 sjj 原环境实际完成的 v2 结果。第 5 节的三个 NumPy 例子仍明确标为构造机制示例；历史主表不参与 v2 结论。

## 1. 优先做什么

**先重跑修正后的共同基线，再优先检验两个已实现的改动：固定教师锚、同教师不同聚合。** 边界良性样本覆盖已纳入所有 v2 方法的数据修复；下表 C 是进一步分离其作用时的单独对照，本轮队列不额外增加这一训练。继续使用 HarmBench ASR、XSTest safe refusal、GSM8K accuracy；不引入新的主指标、训练控制器、路由器或多模型下载。

| 候选 | 最小改变 | 必须固定的对照条件 | 可检验预期与解释 |
|---|---|---|---|
| A：EMA + frozen conditional anchor | `q_mix = 0.75 q_EMA + 0.25 q_frozen`，再计算 `KL(student || q_mix)`；两教师看相同合格 demonstration 和学生前缀 | 对照修正后的 K=1 SDSD；相同数据、步数、学习率、reference KL、种子和评测 | 若 EMA 共同漂移有贡献，预期能力保留或过拒绝改善；ASR 必须实际复核。改善不能自动归因于某一心理学或控制理论定理 |
| B：同 K=3、只换 pooling | 原平均三个 reverse KL，对照先平均三个概率分布再算一次 reverse KL | 完全相同的三个 demonstration、教师权重、rollout、有效 token mask 和温度；保持总蒸馏系数不变 | 若合法回答的相互否决是瓶颈，算术池可能降低过拒绝、保留回答模式。若三个教师本来就几乎一致，预期差异很小 |
| C：辨别性良性样本 | 固定总训练量及 harmful/benign 配比，把 benign 配额中固定 1/4 的容易样本替换为经过审核、含敏感主题但请求合法的样本 | 只改样本覆盖；保持方法、总步数、类别数和损失权重；不使用 XSTest 测试题或其改写 | 若模型学到“敏感词→拒绝”的捷径，预期 XSTest 过拒绝下降；比笼统增加正常样本更直接检验边界覆盖 |

A 的 0.25 是保守的工程起点，没有文献或推导证明它最优。本轮固定该值，不展开权重网格。已有 frozen reference 模型可以复用权重；**不同条件上下文仍需要额外 teacher forward，不能宣称零计算开销**。B 相对于相同 K=3 的原方法不增加教师前向次数。C 可在代码正确、A/B 的方向明确后执行；不要一次叠加三个改动后声称识别了各自作用。

有害训练样本上，A 的 frozen teacher 仍看合格安全 demonstration，避免直接混入未条件化 base 的潜在有害回答；良性样本用合格的有帮助 demonstration。这里利用已有训练标签与示例，不新增推理时分类器。原有 unconditioned reference KL 保留，避免把“换 target”和“删正则”混在一起。

所有对照沿用本轮单种子与同一固定评测集合；诚实报告失败。若更换探索种子，保留原运行记录。由于本轮集合有限，数值变化是该协议下的经验结果，不写成普遍 Pareto 改进的定理。最有价值的结果是相对于修正后的 SDSD，在维持安全收益的同时减少过拒绝、恢复能力；是否达到这一点由实际三列结果决定。

## 2. 多教师首先是聚合规则问题

### 2.1 原 K=3 的平均 KL 是几何池

在一个固定学生前缀 `h` 上，令学生的下一 token 分布为 `p`，冻结的教师分布为 `q_j`，权重非负且和为 1。直接展开 KL 得：

\[
\sum_j w_j D_{\mathrm{KL}}(p\|q_j)
=D_{\mathrm{KL}}(p\|q_G)-\log Z_G,
\quad
q_G(a)=\frac{\exp(\sum_j w_j\log q_j(a))}{Z_G}.
\]

这就是规范化加权几何池，属于 logarithmic opinion pool。它与

\[
q_A(a)=\sum_j w_jq_j(a),\qquad D_{\mathrm{KL}}(p\|q_A)
\]

的算术概率池不同。相反，平均 **forward** KL `sum_j w_j KL(q_j || p)` 与 `KL(q_A || p)` 只差一个与学生无关的常数。不能把不同 KL 方向混为同一多教师方案。

Hinton 等的原始蒸馏论文已经明确区分用算术或几何平均预测分布构造 soft targets；更早的 product-of-experts 工作说明了相乘并归一化所体现的共同约束。因此这里的 pooling 规则是沿用已有思想，新增工作是检验它能否改善 SDSD 安全适配的实测权衡。[1,2]

实现算术池应使用每个教师的概率，或等价的稳定对数形式：

```python
# teacher_logps: [K, ..., vocab], each already log_softmaxed
log_q_mix = logsumexp(teacher_logps + log_weights, dim=0)
```

`softmax(mean(logits))` 仍是几何池，不能用它实现算术池。各教师必须使用相同 vocabulary、温度和学生前缀；教师输出全部 detach，并正确对齐各自不同长度的 demonstration 上下文。

恒等式严格针对固定前缀、固定教师分布成立。它解释 detached rollout 前缀上的 token KL 计算；如果把前缀的采样分布也作为参数化目标求导，`-log Z_G(h)` 不能被无条件当作全训练目标的常数。这里不据此宣称神经网络 on-policy 训练的全局最优或收敛保证。

### 2.2 为什么三个同源拒绝教师未必有帮助

如果 `q_1 = q_2 = q_3`，平均 KL 与一个教师完全相同；**复制相同教师不会因为 K 变大而自动变尖锐**。K 增大可以减少 demonstration 采样噪声，但不能保证消除共同偏差。

更相关的情形是：教师各自偏好不同合法回答，但都给某种拒绝开头较高概率。几何池会下调彼此不认可的合法回答，归一化后共同拒绝占比可能上升。算术池保留各教师给出的合法支持，不要求所有教师都认可同一 token。这是 B 的正向假设；它不等于“几何池一定更危险”或“算术池一定更安全”。共同的小概率有害模式也可能在几何池归一化后变大，算术池也可能保留某个坏教师的有害支持。

统计预测中的 opinion pooling 提供了比“教师越多越好”更准确的语言：需要区分信息来源、校准与分布宽度。Gneiting 与 Ranjan 研究了这些性质，表明线性池可以增加 dispersion，但不自动保持校准或 coherence。我们借用的是聚合机制，不把天气/金融预测结论直接变成 LLM 安全保证。[3]

人类群体估计实验也观察过：共享意见可以压缩多样性而没有相应提高准确性。这只说明“高度一致”与“独立证据多”是不同概念；它不是关于 token pooling 的实验证据，也不能据此推断当前教师的误差相关性数值。[4]

### 2.3 Reverse KL 的表述边界

对固定前缀上可自由选择的完整 categorical 分布，`KL(p || q_A)` 的最优解就是 `p=q_A`。因此“只要 reverse KL 就必然塌到单一模式”不成立。模型族、优化和自采样反馈受限时，reverse KL 可以偏向已有模式；算术池改善的是 target 的可用支持，并不自动改变这些限制。

加入原有 reference KL 后，该局部自由分布问题的解为：

\[
p^*(a)\propto q(a)^{1/(1+\beta)}p_0(a)^{\beta/(1+\beta)},
\]

其中 `q` 可以是 `q_G` 或 `q_A`。这也说明 A 的算术 target 混合与单纯增大 reference KL 并不等价。这个局部推导用于解释设计，不替代真实模型训练。

## 3. 控制机制：EMA 平滑与固定参考承担不同作用

控制论区分系统输出、反馈与外部参考；Åström 与 Murray 的教材提供了这一分析框架。[5] 对本项目，直接由 EMA 定义即可得到：

\[
\bar\theta_{t+1}=(1-\tau)\bar\theta_t+\tau\theta_{t+1}.
\]

它是参数轨迹上的一阶平滑器，恒定输入的稳态增益为 1：如果学生持续停在一个偏离初始行为的位置，EMA 最终也会到那里。减小 `tau` 能减慢跟随，不能凭空提供“应回到哪里”的信息。参数平均的输出并不等于概率平均；这里也不假设它们相等。

Mean Teacher 的原论文在半监督视觉任务中讨论了自生成 target 的 confirmation bias，并用权重平均改善 target；它没有证明 EMA 可以消除所有共同偏差。[6] 冻结模型保留旧任务响应的思路，也已有 Learning without Forgetting 的视觉实验依据。[7] 两者共同支持一个低成本可检验设计：让随学生更新的 EMA 与不随学生漂移的原模型共同提供 target。

**本项目本来已有 frozen reference KL，不能说原方法完全没有固定锚。** A 改的是锚进入目标的方式及强度：固定教师不仅在额外正则项中惩罚偏离，也在 demonstration-conditioned target 中贡献概率。若 A 改善结果，仍需区分“较强的基座保持作用”与“特殊的多教师协同”；本轮不为此扩展成大型超参矩阵。

这个跨领域迁移的可执行内容只有 A；不引入 PID、在线权重调节或额外安全控制器。下文线性 toy 只是展示“平滑”和“固定参考”在一种明确系统中的差别，不是 LoRA 动力学模型。

## 4. 心理学机制：训练相似输入的不同正确反应

Hanson 的原始辨别学习实验让鸽子对不同波长刺激接受不同强化，观察到训练后泛化梯度向远离负刺激的方向移动；Shepard 的工作描述了行为泛化随心理距离变化的规律。[8,9] 可迁移的设计原则是：**训练集应在容易混淆的输入附近，直接提供“何时拒绝、何时回答”的对照经验。** 不能把动物的泛化梯度或心理距离定律当成 LLM embedding 的已证事实。

对 SDSD-Safety，广泛普通问答主要教“明显无敏感内容时回答”，未必教会“提到敏感主题但请求合法时回答”。如果良性构造中过滤掉所有敏感词，模型可以在训练集上同时满足拒绝与帮助目标，却学到错误的主题边界。C 用固定良性配额中的一部分做替换，避免把样本量增长误当作辨别机制的收益。

可使用现有原始训练资源中的合法敏感请求，或离线编写并审核少量意图对照；例如网络安全的防护解释、历史事件分析、求助和预防、包含多义词的普通任务。每条需要审核请求合法性以及目标回答确实有帮助；“含敏感词”只是候选检索条件，不是标签。训练样本、dev 与 XSTest 测试文本分离，记录来源及内容哈希。不要从 XSTest 反向生成几乎相同的训练题。

若 C 只改善普通 dev 而不能减少 XSTest safe refusal，说明这些辨别样本或其覆盖不足；若同样引起 ASR 回升，则报告实际权衡。该预测仍只需要原来的三项评测。

## 5. 三个可复现机制例子

运行：

```bash
python scripts/teacher_pooling_probe.py
```

脚本只依赖 NumPy 与标准库，向 stdout 输出 JSON，不写结果文件。下列结果于 2026-10-30 用 Python 环境中的 NumPy 2.3.5 运行获得。所有概率、偏置和状态转移均为人为构造，不来自 LLM、HarmBench 或真实 teacher logits。

### 5.1 合法回答有分歧，共同拒绝占比上升

两教师分别偏好两个不同合法回答：

| 分布 | help A | help B | refuse | harm |
|---|---:|---:|---:|---:|
| 教师 1 | 0.79 | 0.01 | 0.19 | 0.01 |
| 教师 2 | 0.01 | 0.79 | 0.19 | 0.01 |
| 等权算术池 | 0.40 | 0.40 | 0.19 | 0.01 |
| 等权几何池 | 0.235284385 | 0.235284385 | 0.502959668 | 0.026471561 |

共同拒绝从各教师的 19% 变为几何池的 50.296%；共同有害模式也被归一化放大。算术池保持这里的平均质量。脚本对平均 reverse KL 恒等式的绝对误差为 `2.22e-16`，对同一教师复制三次不改变几何池的最大误差也是 `2.22e-16`。因此应检验的是 **教师分歧落在哪些回答上、如何聚合**，而非用 K 本身解释胜败。

### 5.2 固定锚必须仍承担安全条件

构造 EMA 的 harmful 质量为 0.001，未条件化 frozen anchor 为 0.12。按 A 的 0.25 frozen 权重混合后，harmful 质量变成：

\[
0.75\times0.001+0.25\times0.12=0.03075.
\]

即 0.1% 变为 3.075%。如果条件化 anchor 的 harmful 质量为 0.002，则混合结果为 0.00125。后一个较好的 anchor 分布是例子的输入假设；脚本没有证明条件化一定有效。它支持的工程选择是给 harmful 样本的 frozen teacher 同样安全条件，并真实检查 ASR。

这个 event-mass 等式只针对同一个完整结果空间。固定权重逐 token 混合一般不等于整序列先选教师：`prod_t sum_j w_j q_j(y_t|prefix)` 不等于 `sum_j w_j prod_t q_j(y_t|prefix)`。逐 token 可以组合不同教师的续写，故不能把完整序列混合的有害质量上界直接赋给实际 token mixture。保留这一实现边界即可，不把额外形式证明作为实验前置要求。

### 5.3 降低 EMA 速率与加入固定锚的差别

定义一个刻意简化的标量系统，初值及固定参考 `x_0=0`，学习速率 `eta=0.2`，持续偏置 `b=0.05`：

\[
x_{t+1}=(1-\eta)x_t+\eta[(1-a)(m_t+b)+a x_0],
\quad m_{t+1}=(1-\tau)m_t+\tau x_{t+1}.
\]

`x` 是无约束的抽象标量，不是概率或实际模型指标。运行 2,000 步：

| EMA 速率 tau | 固定锚权重 a | 转移矩阵谱半径 | x 在 2,000 步时 |
|---:|---:|---:|---:|
| 0.20 | 0 | 1.000000000 | 11.123456790 |
| 0.02 | 0 | 1.000000000 | 1.893861454 |
| 0.02 | 0.25 | 0.995289450 | 0.149990710 |

前两行仅改变跟随速度，持续偏置下仍漂移；第三行在该线性系统中趋于有限平衡 `x=(1-a)b/a=0.15`。实际训练还有 reference KL、非线性参数化和变化的数据分布，不能从这个 toy 推出训练稳定性或最优锚权重。

## 6. 贡献归属与下一步判定

本项目可用的论文叙事是：SDSD 的 demonstration-conditioned 自蒸馏能适配安全任务，但初步结果出现过拒绝和能力损失；在修复实现与数据错误后，借鉴既有概率聚合、固定响应保持与辨别训练思想，以少量受控改动检验能否保留安全收益，同时减小这两种代价。

- **沿用**：SDSD 的 on-policy 自蒸馏框架、原 K=3 消融思路、标准概率池与知识蒸馏、EMA 和固定教师思想。[1,2,6,7,10]
- **适配**：合格安全/有帮助示例、条件化 frozen anchor、保持总预算的合法敏感样本替换。OPSA 已有冻结教师及任务条件提示的安全蒸馏，不能声称首次用 frozen teacher 做安全对齐。[11]
- **新增实证**：同样三项评测下，pooling、随时间变化的教师参考和输入边界覆盖各自是否改善安全与效用的权衡。此项目前仍待真实 GPU 实验。

不把 NumPy 例子写进主结果表，也不称其证明 LLM 的实际失败原因。保留当前实验文件、配置、代码版本和小体积摘要即可；无需复制大型 checkpoint 或为这份研究创建独立日志目录。

## 参考来源与使用范围

1. Hinton, Vinyals & Dean (2015). *Distilling the Knowledge in a Neural Network*. [原论文](https://arxiv.org/pdf/1503.02531)，p. 2 明确提到算术/几何平均 predictive distributions 作为 soft targets。支持聚合规则并非本项目新发明。
2. Hinton (2000 technical report; 2002 journal version). *Training Products of Experts by Minimizing Contrastive Divergence*. [作者原文](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)，pp. 1–2 讨论 mixture 与乘积归一化的差别。本文平均 KL 对应权重和为 1 的几何池，不等同于无权重相乘的 PoE；后者复制专家的行为不同。
3. Gneiting & Ranjan. *Combining Predictive Distributions*. [作者预印本](https://arxiv.org/pdf/1106.1638)，引言及 §3 讨论 coherence、calibration、dispersion 和 linear pool。作为统计预测机制来源，不直接外推 LLM 安全效果。
4. Lorenz et al. (2011). *How social influence can undermine the wisdom of crowd effect*. [原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3107299/)。人类数值估计实验，用于说明共同意见和独立证据不同；不用于支持某种 token 聚合公式。
5. Åström & Murray (2008). *Feedback Systems: An Introduction for Scientists and Engineers*. [作者机构资料](https://authors.library.caltech.edu/records/yzs24-xsx88)，[作者电子版](https://www.cds.caltech.edu/~murray/books/AM08/pdf/am08-complete_28Sep12.pdf)。控制反馈与参考的基础框架；本文 EMA 稳态性质和 toy 是显式递推式的直接推导，不是该书针对 LLM 的结论。
6. Tarvainen & Valpola (2017). *Mean teachers are better role models*. [原论文](https://arxiv.org/pdf/1703.01780)，pp. 1–3 讨论自生成 target 的 confirmation bias 与 EMA 权重教师。实验是半监督视觉任务。
7. Li & Hoiem (2016/2017). *Learning without Forgetting*. [原论文](https://arxiv.org/pdf/1606.09282)。通过保存旧网络响应限制遗忘的原始视觉研究，提供固定行为参考的动机。
8. Hanson (1959). *Effects of discrimination training on stimulus generalization*. [DOI](https://doi.org/10.1037/h0042606)，[原论文出版社提供的摘要与首页预览](https://www.researchgate.net/publication/277550747_Effects_of_discrimination_training_on_stimulus_generalization)。访问到摘要及首页；33 只鸽子的辨别训练与梯度位移实验，不声称阅读全文或证明 LLM 机制。
9. Shepard (1987). *Toward a universal law of generalization for psychological science*. [原文摘要](https://pubmed.ncbi.nlm.nih.gov/3629243/)，[大学研究组保存的原文](https://worthylab.org/wp-content/uploads/2018/01/shepard1987.pdf)。心理距离与泛化关系；对 LLM 只作明确标记的设计类比。
10. Nicolicioiu, Pezeshki & Courville. SDSD paper. [原论文](https://arxiv.org/pdf/2606.26091)。用于原方法与 K=3 对照归属；其多样性分析不自动等价于安全保证。
11. OPSA. [原论文](https://arxiv.org/html/2605.15239v1)。冻结教师、任务条件提示与 on-policy safety distillation 的近邻工作；本项目应明确对照，避免重复宣称首次提出。


## 7. v2 实际结果与多教师判定（2026-10-30）

实际完成：26/26 回归（含 Torch 梯度与统计检查）、数据审核/targets、Base + 8 个 100-step run、standard-200/XSTest/GSM8K；逐 run 数值只在 `results/tables/main.md`、`results/tables/ablations.md`、`results/summary.json` 维护。GSM8K 主列已在看到任何重评结果前按 `docs/gsm8k_protocol.md` 切换为公开 exact scorer，旧 strict/flexible 仅作诊断。

- **ASR 测量**：greedy standard-200 中 Base 仅 1/200 harmful，所有方法 0.5–3.0%。该协议没有区分力；不能据此宣布安全等价。固定请求 16-sample ASR@1/4/16 正在按 `docs/asr_protocol.md` 单独测量。
- **SFT/GRPO**：官方 XSTest safe refusal 19.6%/27.6%，flexible GSM8K 72.0%/69.5%；安全拒绝换来严重过拒绝与能力下降。公开 exact GSM8K 均为 0.0%/42.5%，进一步说明格式敏感。
- **SDSD K1 self**：官方 safe refusal 9.2%，unsafe 82.5%，flexible GSM8K 91.0%；公开 exact 3.0%。相对 Base（safe 5.6%，flexible 93.5%，exact 43.0%）是“略多过拒绝 + 格式服从下降”的权衡，不能称无成本。
- **A：EMA+frozen(0.25)**：safe 6.8%，unsafe 79.5%，flexible GSM8K 92.5%，exact 3.5%；与 K3 mean-KL 的 greedy 指标相同。它们是否减少可采样有害输出尚未验证，旧 greedy ASR 不能回答。
- **B：K3 mean-KL vs probability-mean**：greedy ASR 0.50% vs 1.00%，safe 6.8% vs 7.2%，exact 1.0% vs 0.0%，flexible 92.5% vs 91.5%；差异没有配对不确定性支持，不能据此宣称 mean-KL 占优。
- **旧 strict 主列 Pareto 已作废**：它依赖旧 scorer 且把格式改善当成能力。新分析必须同时报告 public exact 与 flexible 诊断，明确规定主指标，并对方法差异给出 request-level 配对区间。

这些 v2 结果只支持一个谨慎结论：拒绝型 SFT/GRPO 的过拒绝代价大，SDSD 系列在 flexible 诊断下接近 Base；公开 exact 主列显示安全适配仍付出格式服从成本。安全采样概率是否真的下降，是 Round 1 的核心测量，不能用旧 greedy ASR 代替。
