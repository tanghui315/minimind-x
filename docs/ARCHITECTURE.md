# MiniMind-X 实现架构说明

版本：v1.4（v1.3 二轮评审修正 + 首批实验结果与 D10/D11 决策，见 §11）
本文回答：每个设计决策（D1–D11）落在哪行代码、每个模块对应任务清单哪一项、怎么跑。

## 定位表述（对外措辞，重要）

MiniMind-X 的 1M 能力是 **million-token externalized working memory**（外置可检索工作记忆）：
长上下文经分块建库，由模型自主学习「何时检索、看多少、记多久」（adaptive memory management），
**不是**全量注意力的 1M long-context understanding。对外表述建议：
「Adaptive Memory Transformer with Million-token Externalized Working Memory」，
避免「1M context model」的说法（该口径会被理解为全注意力长上下文，与实现不符）。

---

# 1. 目录总览

```
model/minimind_x/                 # X 模型包（D1：原 model_minimind.py 一行未动 = Model A baseline）
 ├── config.py                    # T0.1 MiniMindXConfig（preset S/M/L、variant、zone 切分、预算档）
 ├── prelude.py                   # T1.1 Prelude + LocalWindowAttention（T4.4）+ ZoneStack/ZoneBlock
 ├── thinking_core.py             # T1.1 ThinkingCore（循环区）+ MemoryAwareAttention（T5.3）
 ├── coda.py                      # T1.1 Coda
 ├── minimind_x.py                # T0.2 主模型 + generate（T2.4 三模式）+ remap（T0.3）+ encode_cot_steps（T2.6）
 ├── controller.py                # T6.2/T6.3 AdaptiveController：终局编排（建库→初检→预算→循环→再检索）
 ├── losses.py                    # T2.3/T6.4 联合损失 + expected_loops + cot_alignment_loss（T2.6）
 ├── flops.py                     # T0.4 解析 FLOPs/参数量（所有效率指标统一口径）
 ├── router/                      # T2.1 halt / T6.1 retrieval / T5.4 budget
 ├── memory/                      # T5.1 working_memory（含跨轮 K/V 缓存）/ T5.2 memory_manager
 └── retrieval/                   # T4.1 block_store / T4.2 qsa_retriever / T4.3 sparse_attn（Model C）

trainer/
 ├── x_utils.py                   # init/save/weighted_ce/apply_rope_stage/grad_ckpt/tokenize 工具
 ├── oracle_labels.py             # T2.2 R*/B*/retrieve 标签（在线 sweep + 离线公式）
 ├── train_pretrain_x.py          # T1.3 Stage 0（fixed/random/full_unroll 三种 loop 模式，D7）
 ├── train_long_context.py        # T3.3 Stage 1 课程（8K→128K + retrieval-aware 加权）
 ├── train_sft_x.py               # T7.2 SFT-A/B/C（router 课程开关 = Doc 4 §20）
 └── train_grpo_x.py              # T7.3 GRPO + compute penalty + 三相 RL 课程 + 防塌缩

dataset/
 ├── long_context_dataset.py      # T3.2 packing + needle 混合 + gold 加权
 ├── retrieval_qa_dataset.py      # T6.5 {context, question, answer} 检索式样本
 └── mix_dataset.py               # T7.2 配比混合器（30/25/20/15/10）

scripts/
 ├── convert_minimind_to_x.py     # T0.3 权重转换
 ├── run_zone_experiments.py      # T1.5 zone 对比（等 FLOPs）
 ├── eval_depth_dist.py           # T2.5 深度分布 / stop accuracy
 ├── eval_needle.py               # T3.5 needle（backbone / retrieval 两路）
 ├── run_experiment_matrix.py     # T8.1 A/B/C/D × S/M/L 矩阵
 ├── scaling_fit.py               # T8.3 L = A·N^-α·D^-β·R^-γ·M^-δ 拟合
 ├── eval_suite.py                # T8.4 能力/长上下文/效率统一报告
 ├── eval_x.py                    # 推理入口（fast/normal/deep + 终局长上下文）
 ├── launch_h20_8card.sh          # T7.5 8 卡启动
 └── data_gen/                    # T3.1/T5.5：needle/long_dep/multihop/repo/planning/memory_mgmt

tests/                            # 42 项：等价性/轮次轴 KV/损失/memory 不变量/检索 recall/sparse/CoT
```

---

# 2. 设计决策（D1–D9）→ 实现位置

| 决策 | 实现 | 验证 |
|-|-|-|-|
| D1 包结构/原模型不动 | `model/minimind_x/` 新包；`model_minimind.py` 零改动 | test_equivalence（R=1 数值一致 + remap） |
| D2 循环 KV 语义 | 见 §3「轮次轴 KV」（实现期细化的关键点） | test_loop_kv（decode/prefill 一致 + cache 不随 R 膨胀） |
| D3 训练全展开 | `forward(collect_exits=True)` 返回逐轮 exit logits；随机 R 用 `exit_choice` | test_loop_kv.test_exit_choice_selects_loop |
| D4 轮次位置编码 | 每轮复用同一 positions；`loop_embedding` 可选（默认关） | config + test_loop_embedding_forward |
| D5 非可微走 oracle/RL | `controller.forward_with_oracles`（teacher-forcing）；GRPO 阶段自采样 | test_controller_training_forward |
| D6 zone 切分 | `prelude/thinking/coda_layers` + preset S=(4,4,4) M=(8,8,8) L=(10,12,10) | test_config |
| D7 Stage 矛盾统一 | train_pretrain_x `--loop_mode fixed`（默认稳）/ `random`（Doc 3 Phase0 消融） | 冒烟通过 |
| D8 长程 RoPE | `apply_rope_stage` 课程切换重算；主干只覆盖工作序列 | train_long_context 冒烟（两阶段切换） |
| D9 长上下文不进主干 | `encode_context_blocks`（Prelude 分块编码建库）+ controller 全链路 | test_controller_* / eval_needle retrieval 路 |

---

# 3. 轮次轴 KV：D2 的实现期细化（重要）

任务清单 D2 原表述「循环不向 token 轴 KV 追加条目；每轮全量重算」。实现中发现单份 KV cache 的
朴素解读（decode 各轮看 exit 轮历史 KV）与全展开 prefill（每轮自洽重算）**数值不一致**
（tests/test_loop_kv.py 的 decode/prefill 一致性测试最初失败）。修正为文献标准做法
（Huginn 式 latent-depth 轴）：

- **core 层缓存按「轮次」分槽**：`past[core 层] = [(k_r, v_r) for r in 已执行轮]`；
- 第 r 轮的 query 只看历史 token 在第 r 轮的 K/V（历史轮数不足则 clamp 到其最近轮）；
- token 轴每步仍只增加 R 个槽（Prelude/Coda 层照常 +1）；样本停住的那一轮由 hidden gather 决定；
- 该语义下 decode 与全展开 prefill **严格数值一致**（回归测试锁定），B=1 早停场景完全精确；
  batch 内早停的样本会贡献「超出其停住轮的 phantom keys」（batch 级近似，文档化取舍）。

---

# 3.5 与《终局架构方案》（Doc 1）的逐项映射（终局对照审计）

| 终局方案条目 | 实现位置 | 状态/说明 |
|-|-|-|
| §3/§21 数据流（建库→检索→记忆→Prelude→Loop→Coda） | controller + D9 决议 | ✅ 操作顺序取 Doc 2 版（Prelude(问题)→初检→…）；Doc 1 图为外层数据流，矛盾已在任务清单 D9 消解 |
| §4 三段式 Prelude/Core/Coda | zones + config 切分 | ✅ |
| §5-6 Long-Term Memory（1M，不每步参与推理） | LongContextStore（懒编码+LRU，建库一次） | ✅ |
| §7 分层稀疏检索（1M→候选 16~32K→精选） | QSARetriever（coarse 64 块=16K 默认 + fine） | ✅ 默认参数与文档量级一致 |
| §8 Retrieval Attention（QSA 思路） | SparseBlockAttention（Model C）+ 终局走 D9 的 store+检索+记忆 cross-attn | ✅ 功能等价分解（见下方注 1） |
| §9-11 工作记忆 / ADD-KEEP-EVICT-REPLACE / 五因子打分 | WorkingMemory + MemoryManager | ✅ REPLACE=并入新块后按 U 驱逐（add+evict 组合）；Persistence 因子 η 默认 0 可开 |
| §12-13 Dynamic Retrieval + 双 Router（每轮 h_r） | halt/retrieval router + loop_callback | ✅ 决策发生在每轮 pass 之后（loop_callback(r-1, h_r)） |
| §15-16 自适应思考 1~N | max_loop + 早停 generate | ✅ |
| §17 目标 L = L_task + λmCm + λrCr + λdCd | SFT=Doc 2 §18（BCE 监督版，无计数成本）；RL=GRPO reward 三项成本齐（含 Cr=N_retrieval） | ✅ 分阶段实现（见注 2） |
| §18 Unified Controller（memory size / retrieval / depth） | budget/retrieval/halt 三 router + AdaptiveController | ✅ |
| §19 分区注意力 | Thinking Core=Working-Memory Attention ✅；Prelude/Coda 默认 full（可切 local）；「Retrieval Layer」由 D9 路径承担 | ⚠ 有意偏移（见注 1） |
| §20 loop 内不重复全局检索 | store 建库一次，轮间只按需查询 | ✅ |
| §22 训练（curriculum 8K→1M） | M3 到 128K 主干 + 512K/1M 稀疏路径 | ⚠ 可行性修正（v1.1，KV 显存） |
| §23 SFT 数据（Long QA/Repo Debug/Multi-hop/Planning/Agent） | gen_needle / gen_repo_context / gen_multihop / **gen_planning** / mix_dataset | ✅（planning 为终局对照后补齐） |
| §24-25 Reasoning / Memory 数据 | rl_math+code reward / needle+memory_mgmt | ✅ |
| §26 指标（Accuracy/Context/5 项效率） | eval_suite（PPL/效率/FLOPs）+ eval_needle（单针/**多针**/**--data 多跳与长代码**）+ eval_depth_dist | ✅ 多针与数据集口径为终局对照后补齐；公开基准（GSM8K 等）为已文档化边界 |
| §27 创新点 4「统一控制器」 | 三 router 联合训练（SFT-C + GRPO） | ✅ |
| §2「Think Where（区域自适应）」 | 固定 zone（终局定义）+ X1 可选探索 | ✅ 按文档定位为研究扩展 |

**注 1（§19 有意偏移）**：Doc 1 要求 Prelude/Coda 用 Local Attention、中间放 QSA 检索层。在 D9
（长上下文不进主干）下，主干序列只有「问题+生成 token」（短），Local 窗口不再必要，故 Prelude/Coda
默认 full（`prelude_attn/coda_attn="local"` 保留为消融开关）；「QSA 检索层」的读上下文职能由
store 索引 + 检索 + Working Memory cross-attention 分解承担，SparseBlockAttention 作为 Model C
baseline 保留原教旨实现。

**注 2（§17 分阶段）**：Doc 1 的最终目标含检索计数成本 Cr。SFT 阶段按 Doc 2 §18（检索走 BCE 监督、
无计数成本——计数不可微且无 RL 信号时无意义）；Cr 在 Stage 4 的 GRPO reward 中完整出现
（`lam_m*p_m + lam_r*p_r + lam_t*p_t`），即 Doc 1 目标在 RL 阶段完整落地。

---

# 4. 训练流水线（Stage 0–4，D7 编号；Stage 1.5 为 D10 新增）

```bash
# Stage 0 基础预训练（routers 关闭；--loop_mode random 为 Doc 3 Phase0 消融）
# 唯一命令源：scripts/run_stage0.sh（batch/LR/workers 可参数覆盖，D11 吞吐方案由此生效）
bash scripts/launch_h20_8card.sh pretrain        # 或 ./scripts/run_stage0.sh --epochs 2

# Stage 1 长上下文课程（A–C：8K→32K→128K，默认 dense；D/E 512K/1M 走稀疏路径=eval_needle retrieval）
bash scripts/launch_h20_8card.sh longctx

# Stage 1.5 Mid-training（D10，待做：退火 + 数学/代码上采样 + 随机R，A/B 实验）
#   见任务清单 §7.2；数据基建缺口=原始文本配比混合器

# Stage 2 SFT 三相（A: R=1 → B: 随机 R 弹性+一致性 → C: memory/retrieval+router 课程）
bash scripts/launch_h20_8card.sh sft_a && bash scripts/launch_h20_8card.sh sft_b && bash scripts/launch_h20_8card.sh sft_c

# Stage 3/4 GRPO 三相（无惩罚→小惩罚→Pareto；compute penalty 只奖正确轨迹=防塌缩）
bash scripts/launch_h20_8card.sh grpo1 && bash scripts/launch_h20_8card.sh grpo2 && bash scripts/launch_h20_8card.sh grpo3
```

数据：`scripts/data_gen/*` 产 needle / long_dep / multihop / repo / memory_mgmt jsonl；
SFT 混合配比用 `dataset/mix_dataset.py::MixedSFTDataset`（默认 30/25/20/15/10）。

Oracle 标签（Doc 4 §3 self-generated supervision）：`trainer/oracle_labels.py` 提供
`sweep_depth`（在线 R*）、`sweep_budget`（B*）、`sweep_retrieval`（retrieve 标签），
SFT-C 以 `--oracle_rate` 小采样率在线生成（D5 teacher-forcing）。

---

# 5. 推理（三模式 + 终局）

```bash
python scripts/eval_x.py --weight sft_x_c --mode deep          # fast/normal/deep（config.mode_caps）
python scripts/eval_x.py --weight sft_x_c --context_file doc.txt  # 终局：建库→初检→轮间再检索（D9）
```

generate 支持：早停（prefill 按序列 / decode 按步）、fixed_loops、`return_stats`
（avg_loops / memory_steps / re_retrievals，供效率评测与 GRPO penalty）。

---

# 6. 评测与实验矩阵（M8）

| 命令 | 产出 | 对应 |
|-|-|-|-|
| `scripts/run_zone_experiments.py` | dense/full/middle/late_middle 等 FLOPs 对比表 | Doc 3 实验 1–3 |
| `scripts/eval_depth_dist.py` | 难度×loop 分布、oracle R*、stop accuracy | Doc 3 §14 / Doc 6 §10 |
| `scripts/eval_needle.py --mode retrieval` | 深度网格 EM（sparse 路径）；`--needles K` 多针；`--data xxx.jsonl` 直评 multihop/repo/planning（Doc 1 §26 Multi-hop QA / Long Code） | Doc 1 §26 |
| `scripts/eval_suite.py` | PPL + 效率 + FLOPs 统一 JSON | Doc 6 §7 |
| `scripts/run_experiment_matrix.py` | A/B/C/D × S/M/L 启动与记录 | Doc 6 §6 |
| `scripts/scaling_fit.py` | 拟合指数 ± SE、R² | Doc 6 §14 |

Ablation A–F 全部 config 化（T8.2）：A 固定记忆=oracle_budget 钉死；B 一次检索=
oracle_retrieve 全 0；C 固定循环=fixed_loops；D 全层循环=variant full_loop；E 全注意力=
attn 全 full；F 单 router=train_halt/retrieval/budget_router 开关。

---

# 7. 验证状态

- `python3 -m pytest tests/ -q` → **50 passed**（等价性 / 轮次轴 KV / 损失数值 / memory 不变量 /
  检索 recall+MRR / sparse 前向 / CoT 对齐 / 导出往返 / **可学习性**：loop 与 memory 路径
  在固定任务上过拟合且循环区参数确被更新 / **蒸馏**：fine→coarse 梯度确入 query_proj、
  teacher 侧 detach / batch guard / 沙箱拒绝路径）。
- CPU 冒烟全通：train_pretrain_x（loss@R 日志）、train_long_context（阶段切换）、
  SFT-A/B/C（含终局 controller teacher-forcing + 在线 oracle）、GRPO-X（reward/KL 正常）、
  5 个数据生成器、权重转换、eval_x / eval_needle / eval_depth_dist / eval_suite /
  zone 实验 / scaling 拟合 / 矩阵 dry-run。
- 环境：本仓库在 torch 2.3.1 + transformers 5.5.3 下开发验证（模型为纯 nn.Module，
  不依赖 PreTrainedModel 实例化；tokenize 走 `tokenize_tensors` 绕开版本门禁）。
  GPU 训练环境（torch≥2.4）同样适用。

# 8. 已知边界与后续

- **CoT 对齐（T2.6）**：损失（InfoNCE+余弦）与 `encode_cot_steps` 已实现并有测试；接入 SFT-B
  需要 conversations 带 `reasoning_content` 的推理数据切步，属实验期数据工程（未内置管线）。
- **HF 格式导出**：`save_pretrained_x` 落 config.json + model_x.pth（研究格式往返已测）；
  完整 transformers remote-code 导出未做（需发布建模文件，超出研究平台范围）。
- **chunked prefill（T3.4）**：以 SDPA + 梯度检查点覆盖研究规模；>128K 训练按任务清单 v1.1
  走稀疏路径（D9），主干不做 1M 全注意力。
- **基准**：GSM8K/LongBench 等公开基准未内置下载；能力曲线用 eval_depth_dist（算术分级）
  与 eval_needle（合成 needle）作为 lite 口径，公开基准接入属运维工作。
- **memory 管理数据（T5.5）**：生成器已产出 keep/drop 标签；manager 目前以键相似度打分运行，
  文本级 keep/drop 监督的接入点在 `MemoryManager.score`（可替换为学习到的打分头）。
- **检索的可微松弛**：TopK 选择不可微（D5）。已落地两条监督路径：gold CE（needle 数据）与
  fine→coarse 排序蒸馏（v1.3，gold-free，query 侧梯度经 softmax-KL 训练投影，§10-A）。
  gumbel-soft-topk 松弛与 block_hidden 梯度路径（训练 Prelude 编码器）仍列为 V2 实验，
  注意：单纯解掉 controller 的 no_grad 并不能让 query 投影获得梯度（梯度在离散 TopK 处已断）。
- **代码沙箱**：`code_task_reward` 为研究级隔离（RLIMIT + 危险模式静态拒绝，v1.3）；
  不可信输入需容器级隔离（docker/nsjail），属运维工作。

# 9. v1.2 外部评审采纳修正（2026-09-18）

外部 LLM 评审指出的问题中，以下 5 项经核实采纳并修复：

| # | 问题 | 修正 |
|-|-|-|
| 1 | **Budget Oracle 随机标签（真 bug）**：train_sft_x 用 `randint` 当 B* | 接线 `sweep_budget`（逐档 loss → B\*），与 oracle_rate 同门控；未采到标签的批次不监督（router 仍按自身 argmax 执行） |
| 2 | **GRPO off-policy 残余**：`round(avg_loops)` 把轨迹内各 token 不同的 R 压成单值 | 默认改为固定采样 R（fast=1/normal=2/deep=R_max），生成与 logps 重算路径严格一致（on-policy）；`--adaptive_rollout 1` 保留旧行为供消融 |
| 3 | **检索标签 batch 级且无 correctness 分量** | `sweep_retrieval` 升级为 per-sample [B,R] 标签：ΔCE>ε **或** correctness 增益（with 对 without 错）即点亮 |
| 4 | **测试偏接口、缺能力验证与 MRR** | 新增 tests/test_learnability.py（loop 路径/记忆路径过拟合 + 循环区参数确被更新 + loop_embedding 消融可训）；检索测试补 MRR |
| 5 | **定位表述** | 见本文开头「定位表述」：externalized working memory，非全注意力 1M context |

评审中**未采纳**的两条 P0 及理由（供后续讨论）：
- 「loop 必须默认加轮次嵌入」：近年 looped transformer（MoR、Huginn-Loop）默认不加 step
  embedding，Huginn 的消融报告加上反而掉点；轮次信息可由 hidden 状态线性解码。保留为
  `--loop_embedding` 消融开关（trainer 与 zone 实验均已暴露），默认关闭。
- 「打开 retriever 的 query encoder 梯度」：硬 TopK 下该操作是 no-op（梯度在离散选择处已断）；
  真正路径是 §8 所列的 gumbel 松弛 / block_hidden 梯度 / RL 检索质量 reward。

# 10. v1.3 二轮评审采纳修正（2026-09-18）

第二轮外部评审中采纳并修复的 3 项：

| # | 问题 | 修正 |
|-|-|-|
| A | **fine 检索不参与训练**（评审 Bug 3，最有价值一条） | `QSARetriever.distill_loss`：late-interaction 蒸馏 KL(fine ‖ coarse)，teacher detach、student 侧 query 梯度保留；`retrieve` 返回 `cand_idx/fine_scores`；controller 训练前向以 `no_grad=False` 重打分产出 `distill_loss`；SFT-C 以 `--lambda_distill`（默认 0.05）接入。检索投影由此获得 gold-free 的排序监督 |
| B1 | **SFT-C B>1 静默丢上下文**（评审 Bug 1） | 显式 guard：batch>1 直接报错并提示 batch_size=1 + 梯度累积（逐样本 store 的向量化列为 V2 工程） |
| B2 | **代码沙箱研究级即可但未声明/未设限**（评审 P1-3） | RLIMIT（CPU 5s / 内存 512MB / 禁文件写出）+ 危险模式静态拒绝 + docstring 明确「不可信输入需容器隔离」 |
| C | **Memory Manager 措辞**（评审 P1-1） | 全文统一为 adaptive heuristic memory manager（非 learned memory policy；打分头学习化/RL 化在 §8 V2） |

二轮评审中维持不采纳/延后的：P0-1 的「打开 query_proj 梯度」原方案（硬 TopK 下 no-op，蒸馏路径
已实质覆盖其目标）；P0-2「默认开 loop embedding」（保持消融开关，A/B 实验属 Phase 1）；
P1-4 公开基准（Phase 1 实验工作，非代码缺陷）。**结论：代码侧冻结，进入 Phase 1 实验验证。**

# 11. 首批实验结果与 v1.4 更新（2026-09-20）

## 11.1 实验结果（H200-1 8×H200，pretrain_t2t_mini 真实数据）

- **Zone 对比**：8000 步下 middle_loop(3.129) > dense(3.134)（2000 步时 dense 领先——循环优势随
  训练预算显现）；late_middle 被全面支配；middle_loop 为 loss/FLOP 最优点。详见任务清单 §7.1。
- **loss@R 曲线**：两 epoch 稳定 R3 最优后饱和（[1.94, 1.90, 1.89, 1.90]）——最小充分深度成立，
  为 halting oracle 与平均计算节省提供直接依据。
- 数据与权重：out/pretrain_x_768_x.pth；experiments/zone_v2_*.json。

## 11.2 新增决策

- **D10 Mid-training（Stage 1.5）**：采纳社区标准阶段（SmolLM2/Kimi K2 先例），置于架构验证链
  之后做 A/B；独有角度=退火期随机 R 验证「难数据拉大 loss@R 分离度」。配方见任务清单 §7.2。
- **D11 吞吐优化**：下一轮 Stage 0 用 batch 64/卡 × accum 1 + LR 7e-4 + workers 8（26M 模型在
  H200 上显存/利用率双低，预期 2-3 倍吞吐）；run_stage0.sh 参数覆盖即可，无需改代码。

## 11.3 v1.4 工程修复（夜间运行发现，全部已验证）

1. **resume 记账（重要）**：保存语义改为「下一待跑位置」；配置守卫（world_size/batch 变化 →
   当前 epoch 从头）；删除 trainer_utils 跨卡 step 翻倍换算。三场景端到端验证
   （同配置续跑/跨配置从头/已完成退出）。
2. **命令同源**：scripts/run_stage0.sh 为 Stage-0 唯一命令源（night_runner/看门狗/冒烟共用）。
3. **路径锚定**：CKPT_DIR 锚定仓库根绝对路径；SFT/GRPO 补 --tokenizer_path；SFT 加 --max_steps。
4. **夜间运维模式**：21:00 自动启动器（含完整停服序列：vllm-watchdog→容器→清泄漏 worker→等显存
   释放）+ stage0 窗口看门狗 + 早 8 点自动恢复与 morning_report.txt。可复用于后续每夜训练。

## 11.4 D12：五档 Effort + 思考开关（2026-09-20，待实现）

推理五档 off/low/medium/high/max（循环上限 1/1/2/3/4，预算 1K-16K；high=3 依实测 R3 最优）+
`/no_think` 软开关 + SFT-C 档位感知监督（oracle R\* 按档截断、off 混合训练）+ GRPO 五档轨迹采样。
完整设计见任务清单 §7.6；计划随 SFT-C 首跑上线。

## 11.5 D13：Tokenizer 策略（2026-09-20，延后项）

6400 词表实测 fertility（中文叙事 0.95 / 数字 1.36 字符每 token，数字逐位碎片）。验证期保持不动
（架构结论为内部对比）；preset M 重训时与 D10 一并做 32k A/B。详见任务清单 §7.7。
