# SDSD-Safety — 项目进度

## 当前结论（2026-10-30）

已完成修复后的 v2 实现、研究候选和本地统一复核。**原 sjj 环境、GPU、模型与数据缓存均可用；v2 请求意图/回答审核正在 GPU0 运行。** 修复基于远端提交 `0c455fd6b824534e6bb6f606c9b722889a9ee819`；尚未推送。

历史主表与机制数值已移入 `results/legacy_0c455fd.json`，保留原值、来源哈希与失效原因。新主表只接收实际完成的 v2 运行，当前为空。构造分布示例和回归夹具不进入实验结果。

## 已完成与待执行

| 工作 | 状态 | 说明 |
|---|---|---|
| 数据与示范资格修复 | 已实现 | 请求意图独立审核、正常敏感请求保留、split 边界样本预留、统一 targets 与 manifest |
| 训练算法修复 | 已实现 | GRPO 真 KL、dropout、归一化、warmup、原问题保留、严格拒绝与完成记录 |
| 评测与结果来源修复 | 已实现 | standard200 官方模板、Wilson/缺失值、数字解析、旧协议隔离 |
| 多教师与跨领域研究 | 已完成首轮 | `docs/research.md`；条件冻结混合与概率池化开关已实现 |
| CPU 回归与构造示例 | 已运行 | 19 项回归：16 通过、3 因缺少 Torch 跳过；构造示例计算完成 |
| 真模型 smoke | 待执行 | 审核/targets 完成后用 2–4 步小 run 核验教师条件化、KL 梯度、保存与评测 |
| 新数据审核、训练与评测 | 运行中 | v2 审核 GPU0 PID 3079582；之后按 8 个 100-step run 执行 |
| 真实结果替换与 GitHub push | 待训练/评测 | 先本地有效结果，再 commit/push |

## 固定执行顺序

1. 复用原环境执行回归检查，重新审核候选请求/回答，写入同一份已验证 targets。
2. Base + SFT/GRPO/OPSA/SDSD K1，100 updates、seed42。
3. 外部拒绝 K1 与 K3 mean-KL 的同预算消融。
4. K1 条件冻结混合（0.25）与 K3 probability-mean 两个候选。
5. 根据真实三项指标判断是否改善安全—能力取舍；保留有效负结果。

完整命令：在真实 upstream checkout 应用修复后，执行 `QUEUE=all GPU=0 bash scripts/run_main_queue.sh --execute --push`。不具备环境时 runner 记录 `blocked`；`--plan-only` 只记录计划。

## 本次环境

- Python: `/data4/sjj/catelogue/alignment/safety-component/.venv-v31/bin/python`（torch 2.5.1+cu121, transformers 5.8.0, peft 0.19.1）。
- 8×A800-80GB；2026-10-30 11:40 GPU0 空闲 64 GB，GPU4/5/6 各约 51–56 GB 空闲。
- 模型/数据/benchmark 缓存路径与 v2 配置一致，无需下载。
- GitHub SSH 写权限沿用 `jiajunsun68`。

## 验证记录

- HarmBench standard 分类模板与官方 `eval_utils.py` 精确相同；SHA256：`788f4f6aa1491c433c4da76c9140cfc30966cea3ff3875c4d0fcb336d92f60e0`。
- `scripts/teacher_pooling_probe.py` 已实际运行，恒等式误差约 `2.22e-16`；输出明确标为人工构造示例，不是 LLM 实验。
- `python -m unittest discover -s tests -v`：19 项，16 通过，3 项因缺少 Torch 跳过；原环境队列要求这三项实际通过。
- 17 个 Python 文件 AST 解析通过，队列 shell 的 `bash -n` 通过；历史 9 条结果与两份机制 payload 已逐项确认原值未变。
- 本地实际执行 `queue --group all --execute` 于环境检查阶段退出，记录 10 项缺失条件；47 个顺序步骤中 8 个为训练任务，实际启动子任务 0 个、运行 PID 为空。

<!-- QUEUE:START -->
更新：2026-10-30 11:40:32 CST；队列状态：**planned**；GPU：0

| ID | 任务 | 状态 | PID | Exit | 最近输出 / 原因 |
|---|---|---|---|---|---|
| Q01 | Run regression and Torch gradient tests | planned | — | — |  |
| Q02 | Audit request intent and helpful answers | planned | — | — |  |
| Q03 | Verify shared refusal targets | planned | — | — |  |
| Q04 | Generate base_v2 | planned | — | — |  |
| Q05 | Score safety base_v2 | planned | — | — |  |
| Q06 | Score over-refusal base_v2 | planned | — | — |  |
| Q07 | Refresh tables after base_v2 | planned | — | — |  |
| Q08 | Train sft_v2_100_seed42 | planned | — | — |  |
| Q09 | Generate sft_v2_100_seed42 | planned | — | — |  |
| Q10 | Score safety sft_v2_100_seed42 | planned | — | — |  |
| Q11 | Score over-refusal sft_v2_100_seed42 | planned | — | — |  |
| Q12 | Refresh tables after sft_v2_100_seed42 | planned | — | — |  |
| Q13 | Train grpo_v2_100_seed42 | planned | — | — |  |
| Q14 | Generate grpo_v2_100_seed42 | planned | — | — |  |
| Q15 | Score safety grpo_v2_100_seed42 | planned | — | — |  |
| Q16 | Score over-refusal grpo_v2_100_seed42 | planned | — | — |  |
| Q17 | Refresh tables after grpo_v2_100_seed42 | planned | — | — |  |
| Q18 | Train opsa_v2_100_seed42 | planned | — | — |  |
| Q19 | Generate opsa_v2_100_seed42 | planned | — | — |  |
| Q20 | Score safety opsa_v2_100_seed42 | planned | — | — |  |
| Q21 | Score over-refusal opsa_v2_100_seed42 | planned | — | — |  |
| Q22 | Refresh tables after opsa_v2_100_seed42 | planned | — | — |  |
| Q23 | Train sdsd_v2_k1_100_seed42 | planned | — | — |  |
| Q24 | Generate sdsd_v2_k1_100_seed42 | planned | — | — |  |
| Q25 | Score safety sdsd_v2_k1_100_seed42 | planned | — | — |  |
| Q26 | Score over-refusal sdsd_v2_k1_100_seed42 | planned | — | — |  |
| Q27 | Refresh tables after sdsd_v2_k1_100_seed42 | planned | — | — |  |
| Q28 | Train sdsd_v2_external_k1_100_seed42 | planned | — | — |  |
| Q29 | Generate sdsd_v2_external_k1_100_seed42 | planned | — | — |  |
| Q30 | Score safety sdsd_v2_external_k1_100_seed42 | planned | — | — |  |
| Q31 | Score over-refusal sdsd_v2_external_k1_100_seed42 | planned | — | — |  |
| Q32 | Refresh tables after sdsd_v2_external_k1_100_seed42 | planned | — | — |  |
| Q33 | Train sdsd_v2_k3_logpool_100_seed42 | planned | — | — |  |
| Q34 | Generate sdsd_v2_k3_logpool_100_seed42 | planned | — | — |  |
| Q35 | Score safety sdsd_v2_k3_logpool_100_seed42 | planned | — | — |  |
| Q36 | Score over-refusal sdsd_v2_k3_logpool_100_seed42 | planned | — | — |  |
| Q37 | Refresh tables after sdsd_v2_k3_logpool_100_seed42 | planned | — | — |  |
| Q38 | Train sdsd_v2_anchor_k1_100_seed42 | planned | — | — |  |
| Q39 | Generate sdsd_v2_anchor_k1_100_seed42 | planned | — | — |  |
| Q40 | Score safety sdsd_v2_anchor_k1_100_seed42 | planned | — | — |  |
| Q41 | Score over-refusal sdsd_v2_anchor_k1_100_seed42 | planned | — | — |  |
| Q42 | Refresh tables after sdsd_v2_anchor_k1_100_seed42 | planned | — | — |  |
| Q43 | Train sdsd_v2_k3_linpool_100_seed42 | planned | — | — |  |
| Q44 | Generate sdsd_v2_k3_linpool_100_seed42 | planned | — | — |  |
| Q45 | Score safety sdsd_v2_k3_linpool_100_seed42 | planned | — | — |  |
| Q46 | Score over-refusal sdsd_v2_k3_linpool_100_seed42 | planned | — | — |  |
| Q47 | Refresh tables after sdsd_v2_k3_linpool_100_seed42 | planned | — | — |  |
<!-- QUEUE:END -->
