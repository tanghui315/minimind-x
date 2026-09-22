# MiniMind-X V2 残差改造任务清单

依据：《minimind_x_gated_recurrence_loop_mhar_memory_reader_v2.md》（v2.0-design，下称 V2 文档）
版本：v1.1（v1.0 自审修订：补 RL 接线整块 T4.7、6 处内容缺口、2 处验收逻辑收紧；新增 V2 §→任务覆盖对照表）
目标：**终局架构（minimind_x_residual_v2）完整落地**——三个受控模块 + 全部前置缺陷修复 + 契约级验收。

---

# 0. 文档分析结论

## 0.1 V2 在终局架构中的位置

主线不变（检索→记忆→思考），V2 给循环加上三个"受控阀门"，回答三个 previously-open 的问题：

| 新模块 | 回答 | 数学核心 | 位置 |
|-|-|-|-|
| **Loop-MHAR** | 下一轮读哪些历史思考状态 | 按 feature 头在来源轴（E,H1..H(r-1)）softmax 加权 | 每轮 Core 前 |
| **Memory Reader Gate** | 外部证据允许注入多少 | 独立 cross-attn（无 RoPE）+ token 级门 + 有界 scale ρ | Core 首层 self-attn 后、FFN 前，**全模型仅此一处** |
| **Gated Recurrent Update** | 候选状态写回多少 | H(r)=H(r-1)+g⊙(U_r−H(r-1))，g 由 [H_prev,E,ΔU] 低秩 MLP 出 | 整个 Core 后 |

Halt Router **不与 Update Gate 合并**（想多久 ≠ 写多少）。每轮动作序固定：MHAR→Core→Update→Halt→(Retrieval)。

## 0.2 文档的关键纪律（落地时必须遵守）

1. **禁止项**：Update 式不能写成 `H_prev + g*U_r`（重复残差）；bypass 必须是旧 committed 不能偷换 Z_r；softmax 只沿来源轴；Reader Q/K 不用 token RoPE；不能同时零 scale+零 o_proj（双零锁死）。
2. **因果契约**（§10）：检索 query 只见可见前缀；换 gold completion 不得改变初始检索；换未来 token 不得改变当前位置动作。**这不是普通 mask 问题——数据可以经 Router/池化泄漏**。
3. **梯度纪律**（§13）：冻结参数 ≠ no_grad 前向；Reader 投影/检索监督 logits/全部 Router 必须有真实梯度路径；梯度验收按模块分账（区分"应训无梯度/初始化暂无/明确冻结"三种）。
4. **缓存契约**（§11）：三类缓存（Loop History / Memory KV / Token Self-KV）生命周期分离；可变 R 用 R_max 槽 + **per-token last-valid 填充**；append-only past；checkpoint 内禁止副作用。
5. **成本诚实**（§18）：关 Gate 不省已执行的 Core FLOPs；报告 executed vs selected rounds；成本梯度只走动作概率/期望，整数计数只进日志。

## 0.3 代码核验结果（V2 文档附录 C 的 12 项指控全部属实）

| 证据 | 指控 | 核验 | 严重度 |
|-|-|-|-|
| C2 | memory KV 与 token KV 拼接进同一 softmax（RoPE 混用） | ✅ MemoryAwareAttention 确实如此 | 架构级 |
| C3 | get_kv/retrieval_ce 在 no_grad 内 | ✅（v1.3 蒸馏路径已带梯度，gold CE 仍用 detached 分数） | 训练级 |
| C4 | 全展开训练不算 halt 概率 | ✅ halt_fn 仅 halt=True 时设置 | 训练级 |
| **C5/C6** | **SFT-C 检索 query 含 gold 答案（因果泄漏）** | ✅ `prelude_hidden(input_ids)` 全序列池化，input_ids=prompt+answer | **最高** |
| C7 | budget 不联动 topk；memory_size 常量无策略梯度 | ✅ | 训练级 |
| C8 | sparse baseline 块选择∪band 破坏因果 | ✅ `allow = allow_blocks \| band` | 正确性 |
| C9 | SFT/RL 未包 DDP/分片 | ✅ | 多卡 |
| C10 | CLI 默认 768/8 覆盖 preset | ✅ args 非 None 过滤失效 | 工具 |
| C11 | 可学习性测试与真实路径 no_grad 不一致 | ✅（测试已注明） | 测试口径 |
| C12 | cache 按本轮 n_run 重建+clamp，非持久 per-token 契约 | ✅ | 推理级 |

**结论：V2 文档是真实审查了我们代码包的高质量规格，直接作为施工图。**

## 0.4 可复用资产

- 现有三段式/Prelude/Core/Coda、轮次轴 KV、WorkingMemory、检索器、三 Router、多出口监督——全部保留（V2 §3.1）；
- 现有权重链（pretrain_x → sft_x_a/b）作为**迁移 warm-up 起点**（η/update_progress 从 0 升起）；
- V2 附录 A 三个模块参考实现（已过 26 项 CPU 测试）直接移植；
- 夜间自动化、run_stage0.sh、测试框架、D10-D13 决策继续有效（D12 effort 档位与 V2 halt 正交叠加）。

---

# 1. 里程碑与依赖

```
P0 前置缺陷修复（因果泄漏为首，不加新模块就必须修）
 └→ P1 三新模块移植（LoopMHAR / UpdateGate / GatedMemoryReader + 单测）
      └→ P2 Core 集成与输出语义 V2（拆拼接路径、严格兼容模式）
           ├→ P3 缓存契约升级（R_max 槽 / last-valid / 版本轨迹 / 重放）
           └→ P4 训练接线（梯度边界 / warm-up 课程 / loss / oracle V2 / 数据族）
                └→ P5 验收与迁移训练（§21 五组验收 + 旧权重迁移 + 首轮真实训练）
```

关键路径：P0→P2→P4→P5。P1 可与 P0 并行。P3 在 P2 后、P5 前完成即可。

---

# 2. 任务明细

## P0 前置缺陷修复（V2 §3.2，1 周）

| # | 任务（证据项） | 内容 | 验收 |
|-|-|-|-|
| T0.1 | **因果 query 修复（C5/C6，最高优先）** | RetrievalQADataset 输出 `prompt_len` 边界；controller 初始检索只用 `prelude_hidden(input_ids[:, :prompt_len])`；`sweep_retrieval` 改为同前缀反事实：同一可见状态下「REUSE 当前 W」vs「真实 refresh 后 W'」对比（V2 §15.5） | 换 gold completion → 初始检索结果逐位不变；换未来 token → 当前位置动作/logits 不变（因果反例测试） |
| T0.2 | collect_router_logits（C4） | 训练全展开时逐轮计算 halt/retrieval/budget logits（不据此 break），接入 joint_loss | collect_exits 前向返回非空 router logits 且参与 loss |
| T0.3 | budget 联动检索（C7 前半） | `retrieve()` 接受目标容量参数；budget 档位真正改变候选选择 | 库容充足时不同 budget 档产出的 block 集合确实不同（日志 budget_cap vs actual_valid vs unique_blocks；库小于预算时允许相同并记录） |
| T0.4 | sparse 因果修复（C8） | 块选择与因果 band 取**交集**；块选择器因果化（query 只能选含其可见 key 的块） | 未来 token 变化不影响早期位置输出的测试 |
| T0.5 | DDP/分片（C9） | train_sft_x / train_grpo_x 包 DistributedDataParallel（find_unused=True）+ DistributedSampler | 双卡短跑 loss 曲线与单卡一致 |
| T0.6 | CLI 默认 None 化（C10） | hidden_size/num_hidden_layers 等默认改 None，显式值才覆盖 preset/config | preset S 下实测 512×12 |
| T0.7 | gold CE 梯度（C3 剩余） | retrieval_ce 用 grad mode 重算的 coarse logits（对齐 v1.3 蒸馏的做法） | gold CE backward → query/key 投影有梯度 |
| T0.8 | memory_size 常量降级为日志（C7 后半） | joint_loss 的 memory 项移出可训练 loss，保留日志；策略成本走 P4 的期望/轨迹成本 | 常量项不再出现在 backward 图 |

## P1 三新模块移植（V2 §6-8 + 附录 A，1 周，可与 P0 并行）

| # | 任务 | 内容 | 验收（V2 §21.1 不变量） |
|-|-|-|-|
| T1.1 | `residual/loop_mhar.py` | 移植参考实现：全宽 RMSNorm key、共享 pseudo-query（零初始化）、类型偏置（latest=+4）、来源轴 softmax、η 混合 | 单来源恒等；η=0 恒等；权重沿来源轴归一；未来 token 不影响早位置；history 不被原地改 |
| T1.2 | `residual/recurrent_update.py` | 低秩门（3D→64→D）、零 W2/+4 bias 初始化、FP32 门控计算、active mask、force_gate 测试接口 | g=0/1 精确边界；inactive 保持 previous；BF16 有限 |
| T1.3 | `memory/reader.py` | GatedMemoryReader：独立 Q/K/V/O（GQA）、NULL KV 仅对无可见 query、token 级 gate、ρ=tanh(可训练)；`project_memory` **grad mode 契约** | None/M=0/全 masked/混合 batch 空行严格 no-op 无 NaN；scale=0 精确 no-op 但 scale 有梯度；零 W2 初始化后续步内部参数开始更新 |
| T1.4 | 单测移植扩展 | 把参考包 26 项测试范围移植为 `tests/test_residual_invariants.py` / `test_reader_gradients.py` | 全过 |

## P2 Core 集成与输出语义 V2（R-04/05，1-1.5 周）

| # | 任务 | 内容 | 验收 |
|-|-|-|-|
| T2.1 | ThinkingCore 改造 | 每轮：MHAR(history)→CorePass（首层含 Reader）→UpdateGate 提交；halt 读 committed + detached 统计（ḡ、Δ^cand、Δ^commit）；history.append(committed) 不 detach | 顺序与 V2 §12.4 伪代码一致 |
| T2.2 | 拆 memory 拼接路径（C2） | 移除 MemoryAwareAttention 的 KV 拼接；memory 读取全部走独立 Reader（仅 Core 首层）；`config.memory_reader.enabled` 开关 | 全模型只有一处 memory 读取；旧路径以 legacy config 保留供回归 |
| T2.3 | 严格兼容模式 | η=0 + update force=1 + reader scale=0 → 与旧无 memory 路径 logits 逐位一致（迁移回归锚点） | 加载现权重 pretrain_x，兼容模式 vs 旧代码输出 allclose |
| T2.4 | 输出语义 V2 | MiniMindXOutputV2：`core_committed_states` / `coda_exit_states` 分离；`halt_logits`（非概率）；`memory_versions`；`router_trace`；`residual_stats` | Router 一律用 committed state，不用 Coda 后池化冒充（R-05）；生产推理默认不返回完整 history/attention 权重（V2 §12.3，调试显式开启） |
| T2.5 | 每轮动作序固定（V2 §9.1） | Halt 判停后禁止再检索；R_max 轮强制吸收；三动作 REUSE/RESCORE_WARM（无 warm cache 时 mask 掉）/GLOBAL_REFRESH | 动作序测试：不出现"先检索后发现已停" |

## P3 缓存契约升级（R-09/10，1 周，可与 P4 并行）

| # | 任务 | 内容 | 验收 |
|-|-|-|-|
| T3.1 | R_max 固定槽 + per-token last-valid（C12） | 每层保留 R_max 逻辑槽；新 token 填到其实际执行轮，r>r_t 复制最后实际轮 KV；**只填该 token 位置**，不覆盖先前深槽 | 深度序列 4→1→3→2 不丢先前深轮 KV；混合 batch 与逐样本参考一致 |
| T3.2 | prefill fixed_cap 策略 | prefill 用档位上限固定 R_prefill 建 cache；decode 逐 token 自适应；TTFT 与 decode 成本分开记 | 整段 vs 增量数值一致（固定 R 下） |
| T3.3 | memory 版本与轨迹 | WorkingMemoryV2：source_ids 去重、valid mask、memory_version、metadata；轨迹记录 (pos, loop, mem_ver, action, exit)；append-only past | 换 memory 版本后与同轨迹参考重放一致 |
| T3.4 | checkpoint 纯化 | 梯度检查点函数内无检索/无 MemoryManager 修改/无版本变更；控制器先取不可变 payload | 开关 checkpoint 前向/梯度对齐，检索计数不翻倍 |
| T3.5 | Memory KV 缓存键 | cache key 含 request/memory_version/layer/投影版本/encoder 版本/device/dtype（V2 §11.1） | 跨请求/重置/并发不串用 |

## P4 训练接线（R-06/07/11 + §13-16，1.5 周）

| # | 任务 | 内容 | 验收 |
|-|-|-|-|
| T4.1 | 梯度边界修复（R-06） | 离散选择 no_grad + payload 冻结；`project_memory`/检索监督 logits/全部 Router 在 grad mode；13.3 的分模块梯度清单逐项验收；**Memory payload encoder 是否冻结入配置**（V2 §13.1）；可学习性测试改为打真实 controller 路径（C11 测试-现实一致性） | 真实 sft_c_step 中（非绕开 controller 的测试）逐模块梯度表：应训项全有梯度 |
| T4.2 | warm-up 课程（residual_schedule.py） | 同一 run：η 0→1、update_progress 0→1（前 1% steps 0→0.25，1-5% 升满）；严格兼容模式做旧 logits 回归检查点 | 迁移曲线日志（raw gate vs effective gate） |
| T4.3 | loss 接线（V2 §15.2 表） | L_answer（逐出口经 Coda 的 CE，Coda 额外前向计入训练 FLOPs）+ halt/retrieval/budget/retriever（0.1 起步）；**不采用** gate 均值正则/终止-门绑定；**新模块优化器参数组**：1-2× backbone LR，norm/bias/gate scale 不施加 weight decay（V2 §14.4） | loss 配置化；权重表与 V2 §15.2 一致；参数组清单落 manifest |
| T4.4 | Oracle V2 | halt：unresolved 单独标记，不硬标 R_max；retrieval：同前缀反事实（T0.1 基础上）；budget：真差异 sweep（T0.3 基础上）；成本：条件轨迹成本 C(τ) | oracle 生成不读 gold 后缀 |
| T4.5 | 数据族（V2 §16.1 七族） | 无证据/证据必需/无关/相似不适用/新证据纠错/多跳/冲突不足 + **随机事实机制测试**（同模板每次随机换外部事实，train/val 事实组合不重叠）；gold span 按真实 token ID 标注（tokenizer/分块参数变更时重新生成）；**数据划分按文档来源/仓库/任务模板/事实种子切**（V2 §16.4，防同事实换 prompt 进测试集） | 移除/交换 memory → 答案随证据变化 |
| T4.6 | 双执行器（V2 §10.4） | 逐前缀参考执行器（正确性标准）+ 按 memory-version/动作边界分段的批处理执行器；训练用给定动作边界但 query 仅由可见前缀算 | 两执行器语义对应测试 |
| T4.7 | **RL 边界与轨迹概率（V2 §17，v1.1 补）** | ①固定 R token-GRPO 只训 backbone+可微 Gate/Reader/MHAR（现实现已满足），**不宣称优化了 Router 策略**；②轨迹记录 §17.2 全字段（可见前缀/动作 log-prob/memory_version/exit 等）；③联合路径概率 = token log-prob + halt/retrieval/budget 动作 log-prob 之和；④**禁平均深度重放**（[1,3,2,4] 不得用 R=3 整段近似）；⑤完整 Adaptive Policy RL 仅在 T3 重放一致性测试通过后启用（§17.4）；⑥代码沙箱沿用 v1.3 加固 | 轨迹可完整重放；策略 RL 开启前有重放一致性门槛检查 |

## P5 验收与迁移训练（R-12/13/14 + §21，1-2 周）

| # | 任务 | 内容 | 验收 |
|-|-|-|-|
| T5.1 | §21 五组验收 | 数学不变量 / 真实 trainer 梯度 / cache-路径一致性 / 因果与安全 / 任务与效率 | 全部通过并留档 |
| T5.2 | checkpoint/CLI 升级（R-13） | 保存完整 config+架构/缓存版本+optimizer/scaler/各 rank RNG/数据与 warm-up 进度+source/tokenizer manifest+**参数量 manifest（sum(p.numel())）**；CLI 配置继承修复（T0.6 延伸）；missing keys 白名单；**config 校验规则**（V2 §19：D 整除 MHAR heads、query_heads 整除 kv_heads、Reader 插入层存在、budget 受 action mask 限制等） | 严格模式加载无静默冲突；非法 config 启动即报错 |
| T5.3 | 多卡验证（R-14） | P0.5 基础上：DDP 全链路（含 Reader/MHAR/Router unused 参数）| 8 卡短跑收敛正常 |
| T5.4 | 旧权重迁移首训 | 从 pretrain_x warm-up（14.2 课程）→ SFT 重跑（因果修复后）→ 效率统计（executed vs selected、Gate/MHAR/Reader 使用统计） | 基础能力不回退；证据任务有收益；无关/错误证据不放大错误 |
| T5.5 | 效率与风险面板（R-12） | executed_core_rounds vs selected；检索/重投影计数；TTFT/decode 分开；Gate 长期 0/1 的排查流程按 V2 §22.3 表；**flops.py 分项计数改造**（V2 §12.1：reader/gate/MHAR 分项，容量不得代理真实开销；Reader 按 T×M 计不是 M²） | 报告口径与 V2 §18.6 一致 |

---

# 3. 与现有决策/里程碑的关系

- **D10 mid-training**：延后到 V2 模块稳定后（新模块也在退火里训练）；嵌入式域数据集（已传 H200）留作 V2 后的域注入。
- **D11 吞吐 / D12 effort 档位 / D13 tokenizer**：正交，保留；D12 的档位上限作用于 V2 的 R_max/halt 上限。
- **实验基线**：V2 迁移前后必须重跑 zone/loss@R 对照（新模块改变了循环语义）。

# 4. 主要风险（V2 §22.3 摘要）

| 风险 | 处理 |
|-|-|
| Gate 长期≈1 或 ≈0 | 不强制正则；按排查表查覆盖/饱和/零初始化阻塞 |
| MHAR 只读最新 | 与任务表现对照后再判；历史可能真无增益 |
| 旧 memory 路径拆分后初始不等 | 有 memory 时不保证初始相等（V2 §14.3），靠 warm-up+蒸馏对齐 |
| 动态路径训练不稳 | 同前缀反事实监督 + 参考重放，不退回随机标签 |
| 因果泄漏修复后 SFT-C 结果变化 | 预期内（旧结果是泄漏收益），重跑并留对照 |

# 5. 执行顺序建议

1. 第 1 周：P0（T0.1 因果修复最优先）∥ P1 移植
2. 第 2-3 周：P2 集成 + 严格兼容回归
3. 第 3-4 周：P3 ∥ P4
4. 第 5 周：P5 验收 + 迁移首训（夜间窗口）；**策略 RL（T4.7⑤）仅在 T3 重放测试通过后开启，不阻塞主线**
5. 之后：回到 D10 域内 mid-training（嵌入式数据）+ Scaling 矩阵

---

# 6. V2 文档章节 → 任务覆盖对照表（v1.1 新增，审计用）

| V2 章节 | 承接任务 |
|-|-|
| §3.2 前置修复 C2–C12 | T0.1(C5/6) T0.2(C4) T0.3+T0.8(C7) T0.4(C8) T0.5(C9) T0.6(C10) T0.7(C3) T2.2(C2) T3.1(C12) T4.1(C11) |
| §5 统一数学/禁止项 | T2.1、T2.3 |
| §6 Loop-MHAR | T1.1、T4.2（η 课程） |
| §7 Update Gate + §7.5 Halt 特征 | T1.2、T2.1（detached 统计） |
| §8 Memory Reader | T1.3、T2.2、T2.3 |
| §9 动作序/三检索动作 | T2.5（RESCORE_WARM 无 warm cache 时 mask） |
| §10 因果/双执行器 | T0.1、T4.6 |
| §11 缓存契约 | T3.1–T3.5 |
| §12 接口/输出 V2 | T1.x、T2.4 |
| §13 梯度接线 | T4.1 |
| §14 初始化/迁移课程/超参 | T2.3、T4.2、T4.3（LR 参数组） |
| §15 损失/Oracle V2 | T4.3、T4.4 |
| §16 数据族 | T4.5 |
| §17 RL 边界 | T4.7 |
| §18 成本/参数/显存 | T5.5、T5.2（参数 manifest） |
| §19 配置/checkpoint | T5.2 |
| §20 R-01..R-14 | R-01/02/03→T1.1/1.2/1.3；R-04→T2.1；R-05→T2.4；R-06→T4.1；R-07→T0.1；R-08→T3.3；R-09→T3.1；R-10→T3.3+T4.6；R-11→T4.2/4.3；R-12→T5.5；R-13→T5.2；R-14→T5.3 |
| §21 五组验收 | T5.1 |
| §22 风险表 | §4（本文风险表） |

---

# 7. 落地进度（2026-09-22，P0+P1+P2 核心完成）

| 任务 | 状态 | 证据 |
|-|-|-|
| T0.1 因果 query | ✅ | question_len 边界 + controller 因果检索 + 反事实 sweep + test_causality（换 gold 初始检索不变） |
| T0.2 collect_router_logits | ✅ | MiniMindXOutput.router_probs（全展开逐轮 halt/retrieval 概率）+ 测试 |
| T0.3 budget 联动 | ✅ | retrieve(topk_fine=budget_blocks) + budget_cap/actual_valid/unique_blocks 统计 + 测试 |
| T0.4 sparse 因果 | ✅ | 块选择按「块起点≤query 位」因果化 + 掩码取交集 |
| T0.5 DDP/分片 | ✅ | sft/grpo DDP(find_unused)+DistributedSampler；controller 兼容 DDP 包装（self.raw 解包） |
| T0.6 CLI None 默认 | ✅ | 四 trainer hidden/layers 默认 None（preset 不再被覆盖） |
| T0.7 gold CE 梯度 | ✅ | grad mode 重算 coarse logits（对齐蒸馏），backward 入 query 投影（测试） |
| T0.8 memory_size 降日志 | ✅ | joint_loss 不再加常量进 total |
| T1.1-1.4 三模块+测试 | ✅ | residual/loop_mhar、recurrent_update、memory/reader 移植；15 项不变量/梯度测试 |
| T2.1 V2 循环顺序 | ✅ | MHAR→Core(首层 Reader)→Update 提交→Halt（V2 §12.4 顺序） |
| T2.2 拆 memory 拼接 | ✅ | v2 模式 memory 全走独立 Reader（ReaderFirstBlock，唯一读取点）；v1 路径保留 |
| T2.3 严格兼容模式 | ✅ | η=0/λ=0/scale=0 下 V2==V1 逐位一致（测试锁定）——**并抓到真 bug：重写时 v1 轮间链路丢失，已修复** |
| T2.4 输出语义 V2 | ✅ | core_committed_states 与 coda 后状态分离；router_probs 输出 |
| T2.5 动作序固定 | ✅ | 三动作 + RESCORE_WARM mask + halt 后禁检索（见 §7.2/§7.3） |

测试：**73 passed**（原 50 + 因果 4 + 残差不变量 15 + V2 集成 4）。

## 7.1 第二批落地（2026-09-22，P3+P4 完成）

| 任务 | 状态 | 证据 |
|-|-|-|
| T3.1 R_max 槽 + last-valid | ✅ | 槽恒 max_loop；深槽携带自身历史（修复了「填充槽带错槽历史」bug）；浅执行 token 深槽=其最后实际轮；先前深槽不丢（3 项测试锁定） |
| T3.3 版本与轨迹 | ✅ | WorkingMemory 带 source_ids/memory_version（修复统计更新丢版本 bug）；controller 轨迹记录（initial/refresh/reuse+轮次+版本，单调）；投影缓存随内容失效 |
| T3.4 checkpoint 纯化 | ✅ | 开/关梯度检查点：检索计数一致、loss 一致、梯度仍通（测试） |
| T3.5 缓存键 | ✅ | 版本化对象身份失效（测试） |
| T4.1 真实路径梯度表 | ✅ | 真实 sft_c_step（非绕开 controller）逐模块梯度验收（含 v2 三模块/检索投影/三 Router） |
| T4.2 warm-up 课程 | ✅ | residual_schedule.py（η/λ 0→1 线性），pretrain_x/sft_x 接线，冒烟曲线正确 |
| T4.3 LR 参数组 | ✅ | build_optimizer：新模块 2×LR；norm/bias/gate scale/pseudo-query 零 weight decay；多组 cosine×mult 调度 |
| T4.4 halt unresolved | ✅ | halt_bce_loss(valid=) 掩码；不把全错样本硬标 R_max |
| T4.5 七族证据数据 | ✅ | gen_evidence_families.py（七族 + 随机事实 + train/val 事实流隔离）；7000 样本已生成 |
| T4.7 RL 边界 | ✅ | 轨迹全字段 + joint_path_logprob + 禁平均深度门槛（策略 RL 仍待重放一致性门槛后开启） |

测试：**82 passed**（+ 缓存契约 5 + 训练器梯度 4）；本地与 H200-1（torch 2.11）双环境一致。
**今晚（09-22 21:00）自动执行 T5.4 V2 迁移首训**：run_stage0_v2.sh（from pretrain_x，warmup 200 步）
→ SFT-A/B v2；明早 8 点自动恢复服务并出 morning_report.txt。

## 7.2 第三批落地（2026-09-22，T2.5/T3.2/T4.6/T4.7/T5 主体完成）

| 任务 | 状态 | 证据 |
|-|-|-|
| T2.5 三动作空间 | ✅ | RetrievalRouter.action_logits 三分类 + RESCORE_WARM 无 Warm Candidate Cache 时 logits mask + 执行侧拒绝（禁止静默降级全局搜索）；controller 三动作执行（REUSE / RESCORE_WARM 候选重排 / GLOBAL_REFRESH）；Warm Candidate Cache；动作 CE + 可用性 mask；旧 oracle 0/1 仍映射 REUSE/GLOBAL_REFRESH |
| T3.2 prefill fixed_cap | ✅ | generate：prompt 用固定 R_prefill，decode 自适应；TTFT / decode 成本分开记（avg_ttft_loops / avg_decode_loops / executed_core_rounds） |
| T4.6 双执行器 | ✅ | trainer/dual_executor.py：reference_forward_by_prefix（逐前缀参考）+ segmented_forward（按 memory-version/动作边界分段）；对应性与换未来 token 因果测试 |
| T4.7 RL 边界 | ✅ | 轨迹全字段（event/action/round/memory_version/budget/exit/logprob/visible_prefix_len）；joint_path_logprob；assert_replay_not_averaged（禁平均深度重放，均值仅日志）；固定 R rollout 已满足；策略 RL 门槛函数就位 |
| T4.4 halt unresolved 接线 | ✅ | sft_c_step 在线 sweep 时 halt_valid = per_exit_correct.any()；joint_loss(valid=) 真正传入 |
| T5.1 五组验收 | ✅ | tests/test_v2_acceptance.py（任务与效率/因果安全补充）+ 既有 residual/cache/causality/trainer_gradients |
| T5.2 checkpoint/CLI | ✅ | config.validate()（D 整除 MHAR heads、query_heads 整除 kv_heads、sparse 冲突、budget 递增等）；param_manifest(sum p.numel 分模块)；ckpt `__meta__`（arch/cache version + manifest）；训练入口 validate_config |
| T5.5 效率与风险面板 | ✅ | flops.py 分项：reader_attn(T×M 非 M²)/reader_proj/reader_gate/mhar/update_gate；residual 关闭时主干线性于 R 保持；动作计数（reuse/warm_rescore/global_refresh） |
| T5.3 多卡 | ◐ | T0.5 DDP 已就位；8 卡短跑待 H200 夜间窗口验证 |
| T5.4 迁移首训 | ◐ | 今晚 21:00 自动执行（launcher 已武装）；结果见 morning_report.txt |

**审查修复（本批）**：
1. `sweep_retrieval` 在 residual_v2 下误走 v1 `get_kv` 拼接 → 改 `_memory_input` 双路径；
2. `halt_valid` 仅在 losses 有签名、trainer 未传 → 已接线；
3. 三动作标签误入二值 BCE → 拆分（action CE 用 0/1/2，二值 = 是否离开 REUSE）；
4. 检索 recall 测试依赖 RNG 运气（fine 用 Prelude 编码 vs raw-embedding query）→ 收紧到 coarse 契约 + 显式种子；
5. FLOPs 分项在 residual_v2=False 时破坏主干线性 → 按开关计入。

测试：**95 passed**（+ 三动作/双执行器/配置校验/验收汇总 13 项）；本地与 H200 待同步。

## 7.3 第四批落地（2026-09-22，审查缺口收口）

| 任务 | 状态 | 证据 |
|-|-|-|
| T4.6 双执行器真等价 | ✅ | dual_executor 重写：统一 per_token_ce 语义；REUSE 与 GLOBAL_REFRESH 动作边界下 `max\|ΔCE\|≤1e-4`（不再只查形状） |
| T2.1 halt detached 统计 | ✅ | ḡ/Δ^cand/Δ^commit 零初始化旁路进 halt logit（旧权重可加载，初始贡献 0）；测试锁定 |
| T2.4 OutputV2 补齐 | ✅ | `coda_exit_states` / `halt_logits` / `memory_versions` / `router_trace` / `residual_stats` 全部进 MiniMindXOutput |
| T3.3 WorkingMemoryV2 | ✅ | `valid` / `metadata` / `token_valid()`；source_id 去重（同 source 不复制 slot） |
| T5.2 完整 checkpoint | ✅ | save/load/restore：optimizer + scaler + python/numpy/torch/cuda RNG + data/warmup 进度 + tokenizer/source manifest + param_groups 清单；roundtrip 测试 |
| T2.5 动作序测试 | ✅ | halt 后不得再检索（轨迹 round < num_loops_executed） |

测试：**101 passed**（+ field contracts 4 + 动作序 1 + 双执行器 refresh 边界 1）。

剩余待办：T5.3 八卡短跑、T5.4 夜间首训结果验收与对照实验（zone/loss@R、证据任务收益）、策略 RL 仅在重放一致性门槛后开启。
