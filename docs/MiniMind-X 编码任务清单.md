# MiniMind-X 编码任务清单

版本：v1.2（v1.1 全量落地后，进入实验阶段；本版新增首批实验结果与 D10/D11 决策）
日期：2026-09-20

---

# 0. 文档 → 代码的总体映射

| 设计文档 | 落到代码的部分 |
|-|-|
| 终局架构方案（Doc 1） | M6 终局组装：Retrieve→Think→Retrieve 循环、三 Router 统一调度 |
| 数学定义与 PyTorch 模块设计（Doc 2） | M0 包结构、M1/M2/M4/M5 各模块的数学接口与类定义 |
| 自适应潜变量计算架构（Doc 3） | M1 Recurrence Zone、M2 动态深度（第一阶段验证目标） |
| 训练数据与训练策略（Doc 4） | M3 数据构造、M5/M6 oracle 标签、M7 Router 课程 |
| Pretrain SFT RL 三阶段 Recipe（Doc 5） | M7 三阶段训练入口、GRPO + compute penalty |
| Scaling Law 与实验设计（Doc 6） | M8 实验矩阵、ablation 开关、Pareto/scaling 拟合 |

**关键结论：不要按终局架构一次写完。** 文档自身（Doc 3 §17、Doc 4 §20、Doc 5 §24）明确要求课程式推进。
正确的编码顺序是：先在现有 8 层 MiniMind 上验证「Middle Loop + 动态深度」（M0–M2，不依赖长上下文，
用现有数据即可跑），再平行推进长上下文主干（M3），最后组装检索/工作记忆/终局循环（M4–M6）。

---

# 1. 开工前必须锁定的设计决策（P0）

这些决策跨文档未完全定义，但不锁定就无法写代码。以下为推荐方案：

## D1 包结构与 Baseline 保留
- 新增 `model/minimind_x/` 子包（按 Doc 2 §23 布局），`MiniMindXConfig` 继承 transformers `PretrainedConfig`。
- **原 `model/model_minimind.py` 一行不改**，作为 Scaling 文档的 Model A（Dense baseline）；MoE baseline（Model B）复用现有 `use_moe`。
- 新架构通过 `variant` 字段表达：`dense / full_loop / middle_loop / late_middle / minimind_x`（覆盖 Doc 3 §6 实验矩阵与 Ablation D）。

## D2 Loop 的 KV 语义（影响最大的一项）
- **循环轮次不向 token 轴 KV cache 追加条目**：每轮 ThinkingCore 全量重算该轮 K/V；Coda 与后续层只消费「最后一轮」的 KV。
- Working Memory 是**独立的 KV 源**：core 层 attention 拼接 `[causal KV; memory KV]`，不占 token 轴、不进因果 mask。
- decode 时：新 token 走 prelude → core 循环 R 次（attend 上一个 token 的最后一轮缓存 KV）→ coda。
- 这条语义是 generate() 改造与 KV cache 实现的地基，必须先写单测固化。

## D3 训练期循环执行策略（可变循环 × DDP/compile 的矛盾）
- **训练全展开 R_max 轮 + per-exit 监督**：每轮出口都算 logits，得到 `exit_logits[1..R_max]`；「随机 R」（Doc 3 Phase 0 / SFT-B）通过 loss mask/加权实现，不做 data-dependent break。
- 推理期才用 halting router 早停。
- 大规模 pretrain 可切「每 step 采样单 R」省算力模式（config 开关），SFT/oracle 阶段用全展开。

## D4 轮次位置编码
- 每轮复用同一 RoPE positions（循环不推进位置）；可选 learned loop embedding 标识轮次（默认关，作为消融项）。

## D5 非可微操作的处理
- TopK 检索、Evict、预算选择全部不可微。SFT 阶段用 **oracle 标签监督 router**（BCE/CE，Doc 4 §10–14），
  最终由 **RL（GRPO + compute penalty）** 精调；第一阶段不做 straight-through 估计器。
- SFT 期间**执行的动作**（用哪档 memory、哪轮停、哪轮再检索）按 oracle 标签 teacher-forcing 执行；RL 阶段才改为模型自采样决策。

## D6 Zone 切分通用化
- config 用 `prelude_layers / thinking_layers / coda_layers` 三字段（默认 3/3/2 对应 8 层 MiniMind；允许为 0，如 full_loop 变体 = prelude 0 / coda 0）。
- S/M/L preset 按 Doc 6 §4 的 thinking 层数给具体切分：S=12 层 → (4,4,4)，M=24 层 → (8,8,8)，L=32 层 → (10,12,10)。
- 注意：Doc 6 的「100M/500M/1.5B」命名与给定 hidden×layers 的实测参数量对不上（512×12 实测约 40–46M），以 config 实测参数量为准并在实验记录中注明。

## D7 跨文档 Stage/阶段矛盾统一
- Stage 编号（Doc 4 与 Doc 5 编号不一致）：Stage 0 Base Pretrain → Stage 1 Long-Context CPT（8K→1M 课程）→ Stage 2 SFT（A/B/C）→ Stage 3 Adaptive-Compute 联合训练 → Stage 4 GRPO RL。
- **预训练期 Loop 是否开启**（Doc 3 §9 Phase 0 要求「随机 Loop」预训练，Doc 5 §4 要求 Stage 1 loop disabled，互相矛盾）：默认 Stage 0/1 固定 R=1（求训练稳定）；Doc 3 的随机-loop 预训练保留为 config 开关，供消融对比。

## D8 长程 RoPE
- 沿用现有 YaRN 机制，curriculum 每阶段配 rope theta/factor（做成 per-stage 配置）。
- 注：>128K 后主干不再全量 attention（见 D9 与风险表），主干 RoPE 只需覆盖「工作序列」（问题+生成 token），1M 的位置问题由 block store 的块索引承担，而非主干位置编码。

## D9 主干与长上下文的关系（解决 Doc 1 与 Doc 2 的顺序矛盾）★
Doc 1 §3/§21 的流程是「Long Context Encoding → Initial Retrieval → Working Memory → **Prelude** → Loop → Coda」，
而 Doc 2 §1 的数学定义 F(x)=C(T(M(R(P(x))))) 中 **Prelude 在 Retrieval 之前**，两篇直接矛盾。按以下方式统一：

- **1M 长上下文永远不进主干**：prefill 时用一次廉价分块编码（复用 Prelude 权重 + windowed attention，分块处理）把上下文写入 block store（M4）；
- 主干的实际输入序列 =「问题 + 已生成 token」（短序列）；Working Memory 经 core 的 cross-attention 注入，不占 token 轴；
- **初始检索**的 query = Prelude(问题序列) 输出的 hidden（Doc 2 §6.1 的 q=W_q·H 原样成立）；**循环内再检索**的 query = loop hidden state（h_r 更丰富，支撑 Retrieve→Think→Retrieve Again）；
- 单次 forward 的操作顺序采用 Doc 2 版本：Prelude(问题) → Initial Retrieval → Budget 分配 → Loop{Core → halt? → retrieve? → memory update} → Coda。Doc 1 的流程图描述的是**外层数据流**（上下文建库先于一切），两者不再冲突。

## 其他小的不一致（实现时统一）
- Normal 推理模式 memory 档位：Doc 2 §26 写 4K、Doc 5 §28 写 8K → 做成 config，默认 8K / Loop 2。
- Doc 2 §23 模块命名与本文实现的映射：`indexer.py + selector.py` 合并进 `qsa_retriever.py`；`memory_cache.py` 并入 `working_memory.py`（跨 loop K/V 缓存，见 T5.3）。

## 主要风险清单
| 风险 | 对策 |
|-|-|
| R_max 全展开激活显存 ×R_max | per-loop gradient checkpointing（T1.4） |
| oracle performance matrix 是 O(|B 档位|×|R 档位|) 次前向 | 在线小采样率生成 + 离线周期批量生成两种模式（T2.2） |
| 检索监督稀疏 | 合成 needle 数据自带 gold block 标注 → 检索器 CE 监督（T3.1/T4.2） |
| RL collapse（全 R=1 / 全 16K / 全检索） | RL 三相课程 + 最低能力约束（Doc 5 §25，T7.3） |
| 1M block 索引构建成本 | block encoder 用 mean-pool + 线性投影，索引预计算持久化（T4.1） |
| 1M 全量 KV cache 不可行（500M 模型 1M token 的 KV ≈ 98GB，单卡 H20 放不下） | 主干全量 attention 课程只做到 128K；512K/1M 一律走 M4 稀疏路径（store + 检索 + 工作记忆） |

---

# 2. 里程碑依赖图

```
M0 骨架/配置
  ├─→ M1 Recurrence Zone ─→ M2 动态深度 ──────┐
  └─→ M3 长上下文主干 ─→ M4 稀疏检索 ─→ M5 工作记忆 ─→ M6 终局组装 ─→ M7 三阶段 Recipe ─→ M8 Scaling/实验
                                              （M6 同时依赖 M2 与 M5）
```

- **关键路径**：M0 → M1 → M2 → M6 → M7 → M8。
- **M3 与 M1/M2 完全并行**（不同人/不同卡可同时推）。
- M0+M1 约 2 周即可产出第一份研究数据（Middle vs Full Loop，现有数据、8 层、单卡）。

---

# 3. 任务明细

## M0 项目骨架与配置系统（0.5–1 周）

| # | 任务 | 产出 |
|-|-|-|
| T0.1 | `model/minimind_x/config.py`：`MiniMindXConfig`，继承 MiniMind 全部字段，新增 zone 划分、`R_max=4`、halting/retrieval 阈值、memory 档位 {1K,2K,4K,8K,16K}、`block_size=256`、`topk_coarse/fine`、各 loss 权重（α/β/γ/λ）、S/M/L preset、`variant` | config 类 + preset |
| T0.2 | 包骨架 + `minimind_x.py`：`MiniMindXForCausalLM` 组装 Prelude/Core/Coda（M0 阶段 = 普通 block、R=1、无检索无记忆），兼容现有 `generate()` 调用协议 | 可实例化模型 |
| T0.3 | `scripts/convert_minimind_to_x.py`：原 8 层权重 → zone 布局映射加载 | 转换脚本 |
| T0.4 | `model/minimind_x/flops.py`：解析 FLOPs 计算器 f(config, S, R, B_mem, n_retrieval)；后续所有效率指标统一走它 | FLOPs 工具 |
| T0.5 | pytest 基础：前向形状、preset 参数量断言、**R=1 时与原 MiniMind 数值等价**（同权重 logits allclose） | 测试通过 |

**验收**：R=1 等价性测试通过；加载原 pretrain 权重后 loss 连续不跳变。

## M1 Recurrence Zone 固定循环（1–1.5 周）← Doc 3 实验 1–3

| # | 任务 | 产出 |
|-|-|-|
| T1.1 | `prelude.py / thinking_core.py / coda.py`：zone 容器持有连续 layer 切片；`ThinkingCore.forward(hidden, memory_kv=None)` 固定 R 循环，每轮返回 exit hidden | 三 zone 模块 |
| T1.2 | 全展开训练前向：forward 返回 `exit_logits[1..R_max]` + `loop_stats`；`variant` 支持 full/middle/late_middle | 训练化前向 |
| T1.3 | `trainer/train_pretrain_x.py`：复制 train_pretrain.py 骨架；loss = per-exit CE（加权/平均）；支持全展开与采样单 R 两种模式；wandb 记录 loss@R 曲线 | pretrain 入口 |
| T1.4 | per-loop gradient checkpointing | 显存优化 |
| T1.5 | `scripts/run_zone_experiments.py`：Baseline vs Full vs Middle vs Late-Middle，等 FLOPs 预算，输出 val loss + GSM8K 抽样 + FLOPs 表 | 实验脚本 |

**验收**：DDP 下训练稳定无 NaN；拿到「Middle Loop vs Full Loop @ 等计算」第一份对比数据。

## M2 动态深度：Halting Router + Oracle 深度（1–1.5 周）← Doc 3 实验 4、Doc 4 §12

| # | 任务 | 产出 |
|-|-|-|
| T2.1 | `router/halt_router.py`：p_s = σ(W_s·pool(h_r))，per-sequence 二分类（continue/stop） | router 模块 |
| T2.2 | `trainer/oracle_labels.py`：在线 R* 生成 —— 对样本跑 R=1..R_max 各出口 CE/正确性，R* = min{R: loss ≤ min+ε 或 correct}；在线小采样 + 离线批量两模式 | oracle 工具 |
| T2.3 | `losses.py`：L = L_task(exit 加权) + α·L_halt（R* 展开 continue/stop 标签逐轮 BCE）+ β·L_consistency（KL(P_r ‖ P_Rmax)，小深度出口对齐最深出口，target 侧 stop-grad）+ λ·C_compute。**C_compute 必须按 router 概率的期望深度 Σᵣ r·P(stop@r)/R_max 计算——全展开训练下实际执行轮数恒为 R_max，不能用实际轮数**（C_m 同理，按 budget router 概率期望） | 联合 loss |
| T2.4 | 推理早停：**prefill 按序列决策 R；decode 按步（每个新 token 各自决策）**；batch 内统一跑到 max-R、按各样本 halting 决策 gather 各自 exit 的 logits；Fast/Normal/Deep 模式的 R/Memory 上限接线进 generate() | 早停推理 |
| T2.5 | `scripts/eval_depth_dist.py`：loop 分布 vs 合成难度分级、stop accuracy（对 oracle R*）、平均 R、tokens/s | 深度评测 |
| T2.6 | Latent Reasoning Supervision（Doc 3 §10 / 实验 5 / 第一阶段第三目标，v1.0 遗漏）：CoT 数据各 step 用同模型编码为 step 表示，与 loop-r hidden 加对齐损失（余弦/对比），训练「Loop r hidden ≈ Reasoning Step r」，验证 Latent Depth ≈ Reasoning Depth | CoT 对齐 |

**验收**：产出 Experiment 4 数据 —— 等精度下平均计算下降；深度分布与难度正相关；T2.6 的 loop↔step 对齐度可度量并可作消融开关。

## M3 长上下文主干（2 周，可与 M1/M2 并行）← Doc 4 §6–7、Doc 5 §7–8

| # | 任务 | 产出 |
|-|-|-|
| T3.1 | `scripts/data_gen/`：needle 注入器（单/多针，距离 10K/100K/500K/1M，带 gold 位置标注）、长依赖合成（变量定义→远端使用）、document packing（拼 8K~1M）、repo-context 工具（目录+Issue+Patch 模板） | 数据生成器 |
| T3.2 | `dataset/long_context_dataset.py`：按 stage 长度的 packed 数据集（无 padding 浪费），needle 样本带 gold block 标注 | 数据集类 |
| T3.3 | `trainer/train_long_context.py`（Stage 1 CPT）：curriculum 调度器（stage → max_len / rope 配置 / 数据集 / lr）；L = L_LM + λ·L_retrieval-aware（gold/needle token 加权）。**课程范围修正：Phase A–C（8K→128K）走主干全量 attention；Phase D/E（512K/1M）推迟到 M4 完成后走稀疏路径**（全量 KV 在 512K+ 不可行，见风险表） | 长上下文训练 |
| T3.4 | 长序列工程：chunked prefill、activation checkpointing、SDPA 路径在 128K+ 验证 | 工程优化 |
| T3.5 | `scripts/eval_needle.py`：单/多针、深度-位置网格热力图，支持到当前 stage 上限（**>128K 的 needle 评测依赖 M4 的 store 路径**） | needle 评测 |

**验收**：curriculum 各阶段（A–C）val loss 平滑；needle acc 随阶段提升；128K 单卡 H20 可训（batch≥1）；512K/1M 阶段的训练与验收移至 M4 完成后（稀疏路径）。

## M4 稀疏检索与长期记忆库（2 周）← Doc 1 §7–8、Doc 2 §5–6

| # | 任务 | 产出 |
|-|-|-|
| T4.1 | `retrieval/block_store.py`：LongContextStore —— 长文本切块(256 tok)；**块内容** = 复用 Prelude(windowed) 分块编码的 hidden（prefill 时一次完成、可持久化）；**索引 key** = 廉价 mean-pool + 线性投影；core 各层用自身 k/v 投影得到 memory K/V 并缓存（见 T5.3）；支持增量写入、持久化 | 记忆库 |
| T4.2 | `retrieval/qsa_retriever.py`：query 投影 → coarse TopK_c（block 级）→ fine TopK_f → 返回 block KV；gold block CE 辅助监督 | 检索器 |
| T4.3 | `retrieval/sparse_attn.py`：Retrieval Layer —— token attend [local window + retrieved blocks]， retrieved KV 拼入 attention（非因果带 mask） | 稀疏注意力层 |
| T4.4 | prelude/coda 局部（windowed）attention 选项（Doc 1 §19） | 分区注意力 |
| T4.5 | Baseline Model C（sparse attention + fixed depth）配置与入口 | baseline |
| T4.6 | 检索单测：合成 needle store 上 recall@k、延迟、内存占用 | 测试 |

**验收**：needle 任务上 sparse 路径显著省 FLOPs 且精度达标；gold-block 检索 CE 持续下降。

## M5 工作记忆与 Memory Manager（1.5–2 周）← Doc 1 §9–11、Doc 2 §8–9

| # | 任务 | 产出 |
|-|-|-|
| T5.1 | `memory/working_memory.py`：block slots (k, v, meta) 容器；`get_kv()` 供 core 拼接；容量按 block 数计档 | 工作记忆 |
| T5.2 | `memory/memory_manager.py`：U_i = α·Relevance + β·Usage(attention 质量 EMA) + γ·Novelty − δ·Redundancy（+η·Persistence 可选）；add/keep/evict(argmin)/replace，全张量化可 batch | 记忆管理器 |
| T5.3 | ThinkingCore 接入 memory：各 core 层把 memory 块 hidden 用**本层自己的 k_proj/v_proj** 投影成 memory K/V，每次检索后计算一次、跨 loop 复用缓存（Doc 2 §15 的 K_W/V_W 语义）；attention 拼接 [causal KV; memory KV] | 接通记忆 |
| T5.4 | `router/budget_router.py`：memory 档位 CE 分类；oracle B* 生成（performance matrix 的 memory 维） | 预算 router |
| T5.5 | `scripts/data_gen/gen_memory_mgmt.py`：keep/drop/replace 监督数据构造（Doc 4 §19） | 数据 |
| T5.6 | C_m = \|W\|/B_max 正则接入 loss | cost 项 |

**验收**：固定 loop 下 Dynamic Memory ≥ Fixed 8K @ 更低 FLOPs（Hypothesis 1 首证）；evict 不变量单测通过。

## M6 动态检索循环与终局组装（1.5 周）← Doc 1 §12–21 终局架构

| # | 任务 | 产出 |
|-|-|-|
| T6.1 | `router/retrieval_router.py`：p_r = σ(W_r·h)；oracle 标签 = with/without retrieval 的 loss 差（Doc 4 §13） | 检索 router |
| T6.2 | 终局 forward 编排（顺序按 D9）：Prelude(问题) → Initial Retrieval → Budget 分配 → loop{Core → halt? → retrieve? → MemoryManager.update} → Coda；训练全展开 + 各 router 逐轮监督（执行动作按 oracle teacher-forcing，见 D5）；推理逐步决策 | 终局前向 |
| T6.3 | generate() 终局改造：prefill 建一次 store；decode 每步 R 循环 + router 触发的 re-retrieval；**loop 内不重复 global retrieval**（Doc 1 §20）；Fast/Normal/Deep 完整生效 | 终局推理 |
| T6.4 | joint loss 总装：L = L_task + λ₁·L_halt + λ₂·L_retrieval + λ₃·C_memory + λ₄·C_compute；router 课程开关（thinking → retrieval → memory 依次启用，Doc 4 §20） | 总 loss |
| T6.5 | 检索式样本 Dataset（v1.0 遗漏 SFT 侧数据接口）：`{context, question, answer}` schema —— context 分块 tokenize 供建 store（带 gold block 标注），主干序列只含 问题+答案；覆盖 Long Context QA / Multi-hop / 代码仓库任务 | 检索数据集 |

**验收**：端到端长上下文 QA 数值正确、显存可承受；产出 Static RAG vs Dynamic Retrieval 对比（Hypothesis 3）。

## M7 三阶段训练 Recipe 与 RL（2–3 周）← Doc 5 全篇

| # | 任务 | 产出 |
|-|-|-|
| T7.1 | Stage 0/1 入口整合：routers-off 模式 + working-memory-interface 保留开关（Doc 5 §3） | 配置整合 |
| T7.2 | `trainer/train_sft_x.py`：SFT-A（R=1 固定）/ SFT-B（随机 R 弹性）/ SFT-C（memory+retrieval 开启）三相；数据配比混合器（30/25/20/15/10）；在线 dynamic-compute 标签 | SFT 入口 |
| T7.3 | `trainer/train_grpo_x.py`：基于现有 train_grpo + rollout_engine 扩展 —— 轨迹分组加 (memory, loop) 维度；R = R_task − λ_m·P_m − λ_r·P_r − λ_t·P_t；RL 三相（无惩罚→小惩罚→Pareto）；anti-collapse 约束 | RL 入口 |
| T7.4 | reward 函数：math（复用 `safe_math_eval` 判等）、code（subprocess 沙箱测试通过率简版）、长上下文 QA（精确/包含匹配） | reward 库 |
| T7.5 | 新模块的 checkpoint/resume；H20 8 卡启动脚本；DDP + bf16 + compile 验证；`convert_model.py` 支持 minimind-x 导出 transformers 格式；X 模型推理/对话入口（扩展 eval_llm.py 或新增 `scripts/eval_x.py`） | 运维配套 |

**验收**：SFT 三相全跑通；GRPO 在小规模数学任务上 reward 上升且平均 loop/memory 不塌缩。

## M8 Scaling Law 与实验矩阵（2 周 + 持续）← Doc 6 全篇

| # | 任务 | 产出 |
|-|-|-|
| T8.1 | `scripts/run_experiment_matrix.py`：Model A/B/C/D × S/M/L × token 预算统一启动与记录（tokens/GPU/hours/loss/acc） | 矩阵 runner |
| T8.2 | Ablation A–F 全部 config 化并逐一验证可跑 | 消融开关 |
| T8.3 | `scripts/scaling_fit.py`：拟合 L = A·N^−α·D^−β·R^−γ·M^−δ，输出 γ/δ 及置信区间；Compute Pareto 绘图 | 拟合工具 |
| T8.4 | `scripts/eval_suite.py`：能力（GSM8K/BBH-lite/HumanEval-lite/MMLU-lite）+ 长上下文（needle/multi-needle/LongBench-lite）+ 效率（avg memory/retrieval/loop、latency、FLOPs、tokens/s）+ router 指标（depth 分布、stop acc）；统一 JSON + wandb 报告 | 评测套件 |
| T8.5 | 实验记录规范：每次运行落 config 快照 + metrics 到 `experiments/` | 记录规范 |

**验收**：一键复现 Compute Pareto 图与 scaling 拟合；Doc 6 Figure 1–6 所需数据全部可产出。

## 可选探索项（不阻塞终局架构，按需排期）

| # | 任务 | 来源 |
|-|-|-|
| X1 | Adaptive Zone：学习「在哪里循环」（zone 边界可学习/可路由），Doc 3 实验 6 —— 终局架构本身固定 zone，此项为纯研究扩展 | Doc 3 §13 |
| X2 | Agent 基准 lite：ToolBench-lite / 长程合成 Agent 任务，记录 Success Rate + Memory Used + Loop Count + Tool Calls | Doc 6 §18 / Doc 4 §22 |

## 横切任务（贯穿所有里程碑）

- 测试清单（随里程碑同步补）：R=1 与原 MiniMind 数值等价；**prefill/decode 增量一致性**（逐 token decode 与整段 forward 的 logits 一致，D2 KV 语义的回归测试）；memory 驱逐不变量（驱逐前后 get_kv 的 block 集合正确）；检索 recall@k；DDP 单/多卡梯度一致。
- `docs/ARCHITECTURE.md`：最终实现说明，与 6 篇设计文档逐条映射，标注 D1–D9 决策的实现位置。
- 依赖约束：不新增硬依赖（flash-attn 可选）；Python/PyTorch 版本随现有 requirements。

---

# 4. 新增代码总览

```
model/minimind_x/
 ├── __init__.py
 ├── config.py              # MiniMindXConfig（T0.1）
 ├── prelude.py             # T1.1
 ├── thinking_core.py       # T1.1 / T5.3
 ├── coda.py                # T1.1
 ├── minimind_x.py          # MiniMindXForCausalLM + 终局 generate（T0.2/T2.4/T6.3）
 ├── losses.py              # 联合 loss（T2.3/T6.4）
 ├── flops.py               # T0.4
 ├── retrieval/  (block_store / qsa_retriever / sparse_attn)   # M4
 ├── memory/     (working_memory / memory_manager)             # M5
 └── router/     (halt_router / retrieval_router / budget_router)  # M2/M5/M6

trainer/  train_pretrain_x.py / train_long_context.py / train_sft_x.py / train_grpo_x.py / oracle_labels.py
dataset/  long_context_dataset.py（packed 预训练）/ retrieval_qa_dataset.py（检索式 SFT 样本，T6.5）
scripts/  convert_minimind_to_x.py / run_zone_experiments.py / eval_depth_dist.py /
          eval_needle.py / run_experiment_matrix.py / scaling_fit.py / eval_suite.py /
          eval_x.py（X 模型推理入口）/ data_gen/*
tests/    R=1 等价性 / prefill-decode 一致性 / KV 语义（D2）/ memory 不变量 / 检索 recall / DDP 一致性
```

> Doc 2 §23 命名映射：`indexer.py + selector.py` 合并进 `qsa_retriever.py`；`memory_cache.py` 并入 `working_memory.py`（跨 loop K/V 缓存，T5.3）。

原 `model/model_minimind.py`、现有 trainer 与 dataset **保持不动**（Model A/B baseline 直接可用）。

---

# 5. 建议执行顺序（快速胜利优先）

1. **第 1–2 周**：M0 + M1 → 用现有 pretrain 数据 + 8 层 MiniMind 出「Middle vs Full Loop」第一份数据（Doc 3 第一目标）。
2. **第 3–4 周**：M2（动态深度）并行启动 M3（长上下文主干与数据生成器）。
3. **第 5–8 周**：M4 → M5（依赖 M3 的数据与长序列能力）。
4. **第 9–10 周**：M6 终局组装（依赖 M2 的 halting + M5 的 memory）。
5. **第 11–13 周**：M7 三阶段 recipe + GRPO。
6. **第 14 周起**：M8 scaling 矩阵与论文级图表（S 规模先全跑通，再上 M/L）。

总量估算：单人约 13–16 周；关键路径为 M0→M1→M2→M6→M7。

---

# 6. 落地状态（v1.1 编码实施记录，2026-09-17）

**全部编码任务已实现并验证**：`python3 -m pytest tests/ -q` → 46 passed（v1.2 采纳外部评审修正后，
含可学习性/MRR 能力测试，修正记录见 ARCHITECTURE.md §9）；四个训练器、六个数据生成器、
全部入口脚本均通过 CPU 冒烟。实现映射与运行说明见 `docs/ARCHITECTURE.md`。

| 里程碑 | 状态 | 关键产出 |
|-|-|-|
| M0 骨架/配置 | ✅ | config（preset/variant/zone/预算档）、remap 转换、flops、等价性测试 |
| M1 Recurrence Zone | ✅ | 三 zone + 全展开 per-exit 前向 + train_pretrain_x + 梯度检查点 + zone 对比实验 |
| M2 动态深度 | ✅ | halt router、oracle（R\*/B\*/retrieve）、联合损失（含修正的期望 C_compute）、早停 generate、depth 评测、T2.6 CoT 对齐损失 |
| M3 长上下文 | ✅ | 6 个数据生成器（needle/长依赖/multihop/repo/**planning**/memory 管理）、packing+needle 加权数据集、curriculum 训练器（A–C 到 128K）、needle 评测（单针/多针/数据集口径） |
| M4 稀疏检索 | ✅ | LongContextStore（懒编码+LRU）、两级 QSA 检索（gold CE）、SparseBlockAttention（Model C） |
| M5 工作记忆 | ✅ | WorkingMemory（块槽+跨轮 KV 缓存）、MemoryManager（五因子打分/驱逐）、budget router |
| M6 终局组装 | ✅ | AdaptiveController（建库→初检→预算→循环→轮间再检索）、teacher-forced 训练前向、终局 generate、检索式数据集 |
| M7 三阶段 Recipe | ✅ | SFT-A/B/C（router 课程）、GRPO-X（compute penalty + 三相 RL + 防塌缩 + math/code reward）、混合配比数据集、H20 8 卡脚本、研究格式导出 |
| M8 Scaling/实验 | ✅ | 实验矩阵 runner、ablation A–F config 化、scaling 拟合（±SE/R²）、eval_suite 统一报告 |
| 横切 | ✅ | 42 项测试、ARCHITECTURE.md、零新增硬依赖 |

**实现期对 D2 的细化（重要）**：单份 KV cache 的朴素「循环不追加 KV」语义会导致 decode 与
全展开 prefill 数值不一致（测试最初失败）。已细化为「轮次轴 KV」：core 层缓存按轮分槽、
第 r 轮只看历史第 r 轮的 K/V，decode/prefill 严格一致（Huginn 式 latent-depth 轴；
token 轴每步仍只 +R 槽）。详见 ARCHITECTURE.md §3 与 thinking_core.py 头注。

**边界与后续（不阻塞）**：CoT 对齐的数据管线（reasoning_content 切步）、HF remote-code 完整导出、
公开基准（GSM8K/LongBench）接入、memory 文本 keep/drop 监督头，见 ARCHITECTURE.md §8。
终局方案（Doc 1）的逐项对照审计表见 ARCHITECTURE.md §3.5（含两处有意偏移的说明：§19 分区注意力
在 D9 下的等价分解、§17 目标的 SFT/RL 分阶段实现；planning 数据与多针/多跳评测为对照后补齐）。

---

# 7. 首批实验结果与新增决策（2026-09-20，H200-1 8×H200 实测）

## 7.1 实验结果

**Zone 对比（Doc 3 第一目标，pretrain_t2t_mini 真实数据）**

| 预算 | dense R1 | middle_loop R2 | full_loop R2 | late_middle R3 | 结论 |
|-|-|-|-|-|-|
| 2000 步 | **4.4623** | 4.5419 | 4.5521 | 4.6079 | 短预算 dense 最优（文献已知：循环需更长训练） |
| 8000 步 | 3.1343 | **3.1290** | 3.0823(2×FLOPs) | 3.1844 | middle_loop 反超 dense；late_middle 被支配；middle 为 loss/FLOP 最优点 |

**loss@R 深度曲线（middle_loop 全展开训练，核心假设信号）**

- epoch 1 终点：[1.94, 1.90, **1.89**, 1.90]（R3 最优后饱和，epoch 2 复现同形态）
- 「最小充分深度」成立：训练后的模型用第 2-3 轮循环降 loss，第 4 轮无增益——
  这为 halting oracle（R\* 集中在 2-3）与 compute 节省（平均 R≈3 vs R_max=4）提供直接依据。

## 7.2 D10：引入 Mid-training（Stage 1.5，2026-09-20 决策）

社区依据：mid-training 已成标准阶段（survey 已出）；Kimi K2 用其做领域注入+长上下文扩展；
SmolLM2（1.7B，同体型）三阶段中 Stage 2 退火期数学上采样/低质下采样收益明确。

**采纳但延后**：放在架构验证链（SFT-C + 首轮 GRPO）跑通之后，作为独立 A/B 实验，不插队污染对照。

- 位置：Stage 1（长上下文课程，本身就是 mid-training 的长上下文分支）与 Stage 2（SFT）之间；
- 数据（~2 亿 token）：60% 通用高质量（pretrain_t2t 完整版抽样）+ 25% 数学（data_gen 合成+开源）
  + 10% 代码（repo 生成器）+ 5% 8K 长序列打包；
- 调度：WSD 式退火 5e-5→5e-6；
- **独有角度：退火全程随机 R∈[1,4]（Doc 3 Phase 0 设计）**——验证「难数据退火拉大 loss@R 分离度」，
  若成立则 SFT-B 弹性与 halting 监督信号同步增强，是可写入论文的实验点；
- 验证：A/B（有/无 mid-train）→ 各接 SFT-A/B → 对比 loss@R 分离度 + 数学题准确率；
- 基建缺口：原始文本格式配比混合器（现 mix_dataset.py 为 SFT 格式）——约一天工作量，下周做。

## 7.3 D11：吞吐优化（下一轮 Stage 0 重跑时生效）

26M 模型在 H200 上仅用 9.5GB/141GB 显存、利用率 53-70%（kernel/数据加载瓶颈）。
下一轮（明晚起）：batch 64/卡 × accum 1（有效 512）+ LR 7e-4（√n 缩放）+ num_workers 8。
预期 2-3 倍吞吐。run_stage0.sh 已支持参数覆盖，无需改代码。

## 7.4 工程修复记录（2026-09-20，全部已验证）

| 修复 | 内容 |
|-|-|
| resume 记账 | 保存语义改为「下一待跑位置」（epoch 末存 epoch+1/step 0）；跨 world_size/batch 配置守卫（变化则当前 epoch 从头）；删除 trainer_utils 错误的跨卡 step 翻倍换算。三场景端到端验证 |
| 命令同源 | scripts/run_stage0.sh 为 Stage-0 唯一命令源；night_runner/看门狗/冒烟全部引用（杜绝首晚两次「命令不同源」事故） |
| 路径锚定 | checkpoint 目录锚定仓库根绝对路径（原 ../checkpoints 在仓库根运行时落项目外）；SFT/GRPO 补 --tokenizer_path |
| SFT max_steps | 大数据集夜间限量跑（--max_steps） |
| 夜间运维 | 停服完整序列（vllm-watchdog→容器→清泄漏 worker→等显存释放）；21:00 自动启动器 + stage0 窗口看门狗 + 早 8 点自动恢复与报告 |

## 7.5 修订后执行顺序（替代 §5 第 1-2 周条目）

1. ✅ zone v1/v2 + Stage 0 两 epoch（已完成，见 7.1）
2. 本周：SFT-A/B（夜间自动）→ **SFT-C（终局 controller + 蒸馏 + oracle）→ 首轮 GRPO**——架构验证闭环
3. 下周：Stage 1.5 mid-training A/B（D10）+ 长上下文课程（M3）+ 吞吐优化重跑（D11）
4. 之后：Scaling 矩阵（S 规模全链 → M 规模）

## 7.6 D12：五档 Effort 配置 + 思考开关（2026-09-20 决策，待实现）

对标社区：OpenAI o 系 reasoning_effort（low/medium/high）、GLM-4.5 thinking budget、Qwen3 /think //no_think。
我们的优势：档位直接约束循环上限（比提示词方案硬），且 R=1 路径自 Stage 0 起受 per-exit 监督、
SFT-A 整阶段专训——「关思考」= 真的只跑 1 轮（省 75% 计算），非 CoT 剥离式。可作论文对比点。

**① 推理五档**（config.mode_caps 扩展，5 分钟改动）：

| 档位 | 循环上限 | 记忆预算 | 语义 |
|-|-|-|-|
| off | 1 | 1K | 关闭思考（=dense 直通，等价性测试保证） |
| low | 1 | 2K | 简单事实问答 |
| medium | 2 | 8K | 默认档 |
| high | 3 | 16K | 实测 R3 最优点（跟数据走，非凑满 4） |
| max | 4 | 16K | 全力 |

fast/normal/deep 保留为别名。

**② 对话软开关**（eval_x 入口层，10 分钟）：解析用户消息中 `/no_think`、`/think <档位>` → 覆盖当轮
effort；模型零改动（Qwen3 式）。

**③ 档位感知训练**（SFT-C 增强，约 20 行，核心新增）：
- 每 batch 采样档位，oracle R\* 按档位上限截断（R\*=min(R\*,cap)），halt 标签随截断——router 学会
  「预算内选最优深度」；
- off 档混合训练：部分 batch 强制 R=1 且无 halt 监督（关思考场景质量不崩，Qwen3 混合数据做法）；
- GRPO mode_diversity 从三档循环升级为五档采样，compute penalty 按档位归一。

**接线点**：config.py mode_caps / train_sft_x.sft_c_step oracle 截断 / train_grpo_x MODE_CYCLE /
eval_x.py 软开关 / tests 档位映射与 off=R=1 等价。
**计划**：与 SFT-C 首跑一并上线（架构验证链最后两环前插入，不阻塞）。

## 7.7 D13：Tokenizer 策略——验证期保持 6400，M 规模时 32k A/B（2026-09-20 决策，延后）

现状：MiniMind 自训 byte-level BPE，词表 6400，Qwen 式对话模板 + `<think>` 标签（D12 软开关免改）。

**实测 fertility（2026-09-20）**：中文叙事 0.95 字符/token（现代 15 万词表约 1.5-1.6）、中文技术 1.91、
英文 3.06、代码 2.54、数学/数字 1.36（长数字逐 1-2 位碎片化）。三个影响：有效数据量缩水 ~40%、
数字碎片压低 needle EM 与 math reward 天花板、长上下文按字符计缩水。

**决策**：
- 架构验证期（当前）：**保持 6400 不动**。所有架构结论（zone/loss@R/halting/memory）为同 tokenizer
  内部对比，不受词表影响；换词表 = 全部权重作废。
- 参数经济学约束：6400×768=4.9M（占 26M 模型 19%，tie 共享只计一次）；32k 词表 embedding 单项
  24.6M > 模型其余部分——768 宽度下不可行，**至少 preset M（1024 宽）才划算**。
- 触发条件：升级 preset M 重训时，与 D10 mid-training 一并做 32k 词表 A/B（重点看数字 3 位分组对
  needle/math 的天花板抬升）。
- 论文口径：报告 fertility 并说明内部对比有效性。
