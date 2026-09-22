# MiniMind-X V2：受控循环、历史状态路由与独立记忆读取

## Gated Recurrence + Loop-MHAR + Memory Reader Gate 落地技术方案

**版本：** v2.0-design  
**日期：** 2026-09-21  
**实现基线：** `minimind_x_code_bundle(2).md`，55 个新增代码文件、5083 行代码。  
**文档性质：** 针对既有 MiniMind-X 的增量架构与实现规格，不是已经完成训练的技术报告。  
**交付边界：** 本文包含完整设计、代码改造清单、训练接线、缓存契约、验收标准，以及三个核心模块的独立 PyTorch 参考实现。参考实现已做 CPU 单元测试；尚未与原仓库合并，不代表完整解码器、多卡训练或 1M 长上下文已经验证。  
**基线文件 SHA-256：** `d94753134740998e40e88eee0c8be1d164892ea125aea0e4b3cd096af50f0da1`

---

## 阅读导航

- 第 1–3 节：最终决策、研究依据与当前实现的衔接。
- 第 4–8 节：统一张量与数学定义，三个新增模块的准确职责。
- 第 9–12 节：动态检索、因果性、KV 缓存、PyTorch 接口和执行流程。
- 第 13–17 节：训练梯度、初始化、损失、数据与 RL 接线。
- 第 18–22 节：成本、配置、开发任务、验收与上线条件。
- 附录 A–D：实现参考、测试范围、证据索引与参考资料。

# 1. 最终决策

## 1.1 保留已经确定的终局主线

MiniMind-X 的主线不变：

```text
长上下文/领域知识源
        ↓
Dynamic Retrieval
        ↓
Adaptive Working Memory
        ↓
Adaptive Thinking
        ↓
答案
```

本次改造只进一步回答三个问题：

| 新模块 | 回答的问题 | 作用位置 |
|---|---|---|
| Loop-MHAR | 下一轮应读取哪些历史思考状态？ | 每轮 Core 执行前 |
| Memory Reader Gate | 外部知识读入后，允许注入多少？ | Core 首层 Self-Attention 之后、FFN 之前 |
| Gated Recurrent Update | 本轮候选状态，真正写回多少？ | 整个 Core 执行后 |

**Halt Router 仍然单独存在**，决定是否执行下一轮；它不与 Update Gate 合并。

最终推荐结构为：

```text
                              长上下文 / Domain SLM Slots
                                          │
                                    动态检索/桥接
                                          │
                                   Working Memory
                                          │
输入 → Embedding → Prelude → 固定 Anchor E │
                            │             │
                            ▼             │
                       H^(0) = E          │
                            │             │
                ┌───────────▼─────────────┴─────────────┐
                │              Thinking Loop r          │
                │                                       │
历史 [E,H1,...] ─► Loop-MHAR → 本轮读取状态 Z_r           │
                │                  │                    │
                │        首层 Self-Attention             │
                │                  │                    │
                │        独立 Memory Cross-Attention ◄───┤
                │                  │                    │
                │          Memory Reader Gate           │
                │                  │                    │
                │       FFN + 其余共享 Core 层           │
                │                  │                    │
                │          候选状态 U_r                 │
                │                  │                    │
上一轮 H^(r-1) ─► Gated Recurrent Update ◄── Anchor E    │
                │                  │                    │
                │          提交状态 H^(r)               │
                └──────────────────┬────────────────────┘
                                   │
                      Halt Router：停止 / 继续
                           │               │
                         停止            继续
                           │               │
                           │       Retrieval Router
                           │       复用 / 重排 / 全局检索
                           │               │
                           │       更新 Memory 后下一轮
                           ▼
                          Coda → LM Head
```

这里的循环单位仍然是完整共享 Core，例如 `[L4 → L5 → L6]`，不是某个 Attention 层自己重复，也不是每个 Loop 都重新编码全部长上下文。

## 1.2 本版本明确不增加的东西

本版本不引入全模型 mHC，不把所有层替换为原版 Full AttnRes，不同时增加 GDN/KDA，也不要求领域小模型与大模型共享 tokenizer。

领域小模型的 Memory Slots 只作为另一种 `MemorySource` 接入。本文不把“训练领域小模型”和“训练跨模型 Bridge”混入残差改造的核心交付。

这是一套完整目标架构。训练中的 warm-up、兼容性测试和修复清单，都是保证该架构可训练的工程步骤，不是重新做一轮架构筛选。

## 1.3 必须校准的五条理解

1. **无显式 loop embedding，依然是合法的 recurrent/weight-tied Transformer。** 不能把 `loop_embedding=False` 本身定为错误。本文保留该开关，默认关闭。
2. **Update Gate 小，不等于模型已经答对，也不等于一定该停止。** 可能是正确状态趋稳，也可能是错误固定点、门控饱和或欠训练。
3. **算完候选结果后再关 Gate，不会省掉已经执行的 Core FLOPs。** 真正省计算需要 Halt、样本压紧或上游跳过操作。
4. **有 Gate 不保证永远不会 overthinking，也不保证通用能力绝不下降。** 它提供一种可学习的状态更新机制，效果需要任务与数值验收。
5. **Loop-MHAR 是本文提出的“将多头深度路由迁移到 loop 边界”的设计。** 已有 MHAR 论文结果不能直接当作 MiniMind-X 上这一组合的实证结果。

# 2. 研究依据、借鉴部分与自定义部分

## 2.1 已核实的研究依据

| 来源 | 本文借鉴的内容 | 不直接推导的结论 |
|---|---|---|
| Attention Residuals，arXiv:2603.15031 | 使用归一化来源表示与学习 query，对历史深度来源做 softmax 加权读取；存在 Block 版本 | 不据此保证任何 loop 历史混合都优于普通残差 |
| Multi-Head Attention Residuals，arXiv:2607.27230 | 将 feature width 划分为多个头，每头独立选择深度来源 | 论文主要操作 layer/sublayer 来源；不是本文的 committed-loop-state 历史 |
| Gated Recurrent Transformers / RecurrentGPT，arXiv:2608.15062 | Prelude–shared Core–Coda 结构、逐元素更新门、固定 Prelude 条件 | 本文省略原工作中的状态/门控噪声，门控网络输入也不声称逐行复现原实现 |
| mHC，arXiv:2512.24880 | 多路残差与约束混合是另一条可行路线 | 本文不实现 mHC，也不据其数值稳定结论保证本方案稳定 |
| PyTorch 官方 Autograd / SDPA 文档 | 正确区分冻结参数与 `no_grad`；明确 attention mask 和 dropout 语义 | 框架支持 API，不等于目标硬件一定使用最快 kernel |

资料与链接见附录 D。[R1–R6]

对 arXiv:2608.15062，本次检索的 arXiv 摘要索引与作者模型卡在标题表述上存在 `RecurrentGPT` / `Gated Recurrent Transformers` 的差异；本文以论文编号识别来源，以作者模型卡确认相关设计，不把标题差异扩展成两个独立方法。[R3]

## 2.2 本文实际新增的组合设计

以下均是 MiniMind-X 的实施选择，而不是宣称来自某篇论文的原样方案：

- 用完整的 **committed loop state**，而非所有 Attention/MLP 原始输出，作为 Loop-MHAR 的来源。
- 只在 loop 边界执行一次历史读取，默认共享一组四头 pseudo-query。
- 给最新状态一个可学习初始偏置，避免迁移时一开始平均混合全部历史。
- 仅在 Core 首层增加一次独立 Memory Reader，避免重复多层读取扩大成本。
- Update Gate 使用上一轮状态、Prelude Anchor、候选差分三类信息，通过低秩 MLP 输出逐通道门。
- 明确区分 token self-KV、memory KV 和 loop history，制定训练/推理各自的生命周期。

这些选择形成一套可实现的方案，但“优于其他架构”“减少幻觉”“收益叠加”仍是待验证的目标。

# 3. 现有实现的衔接与必须修复的前置条件

## 3.1 可以直接保留的部分

基线已经有 Prelude/Core/Coda 切分、共享 Core、分轮次 self-KV、WorkingMemory 容器、检索与预算控制器以及多出口监督接口。[C1–C4]

建议保留这些结构和原有权重名称，仅通过新增模块及明确的版本字段升级。

## 3.2 为什么不能只“再加三个类”就开训

基线代码存在一些与本次改造直接相关的接线问题。这里是基于附件源码的判断，不是沿用此前评审中的成熟度评分。

| 位置 | 附件中可见的情况 | 本版本的处理 |
|---|---|---|
| `MemoryAwareAttention.forward` | memory K/V 与 token K/V 拼接，进入同一个 softmax | 拆成 Self-Attention 与独立 Reader；禁止新旧两条 memory 路径同时读入 |
| `AdaptiveController.forward_with_oracles` | `get_kv()` 位于 `no_grad` 范围内 | 冻结索引/编码与训练 Reader 投影分开；训练 KV 投影保留计算图 |
| `retrieve_into_memory` → `retrieval_ce` | 在 `no_grad` 生成 coarse scores 后直接用于 CE | 有监督时重新在 grad mode 计算 logits；不能只把 detached loss 加进总 loss |
| `collect_exits=True` 的训练前向 | Halt 概率只在 `halt=True` 时产生，训练时可能返回空列表 | 新增独立 `collect_router_logits`，训练算概率但不据此提前 break |
| SFT-C 的 query 构造 | `input_ids` 含问题和 gold 答案，Prelude 后对全序列池化用于检索 | query 必须只来自当时可见前缀；不能让 gold 后缀影响前面的 memory 选择 |
| `sweep_retrieval` 调用 | 调用时未传 memory，with/without 两条路可同为无 memory | 改成同一前缀状态下“复用当前 W”与“真实 refresh 后 W'”的反事实比较 |
| Budget 与 Top-K | budget 控制淘汰容量，但检索仍使用固定 `topk_fine` | 将目标容量/缺口真正传给候选选择；否则多数预算档可能实际取到相同内容 |
| Memory cost | `memory_size` 是新建常量张量 | 常量用于日志；策略优化用可微期望成本或动作 reward |

依据：[C2–C7]。上述问题不会因增加 Gate 自动消失。

另有两个上线前必须阻断的基础问题：基线 SparseBlockAttention 的 `allow_blocks | band` 之后没有统一再与因果 mask 取交集，块级选择本身也需要因果限制；SFT/RL 启动了 distributed 环境不等于已经正确包装 DDP 并分片数据。[C8–C9] 本文不把修复这些问题包装成残差创新，但训练前必须处理。

# 4. 统一符号、三类记忆与张量契约

## 4.1 符号

| 符号 | 含义 | 典型形状 |
|---|---|---|
| B | 当前批次样本数 | 标量 |
| T | 当前处理的 token 数，prefill 可大，decode 通常为 1 | 标量 |
| D | MiniMind-X hidden width | 读取实际 config，不写死 |
| R | 本次最大 Core 循环数 | 默认 4 |
| E | 同一前向、同一 token 的 Prelude Anchor | `[B,T,D]` |
| H^(r) | 第 r 轮提交后的状态 | `[B,T,D]` |
| U_r | 第 r 轮完整 Core 给出的候选状态 | `[B,T,D]` |
| Z_r | 历史混合后，本轮 Core 的读取状态 | `[B,T,D]` |
| H_hist | Loop-MHAR 的 feature heads 数 | 默认 4，要求 D 可整除 |
| M | 当前活跃 memory token/slot 数 | 可变，不等于 T |
| D_m | memory payload 的宽度 | 可与 D 不同 |
| W | memory payload | `[B,M,D_m]` |
| valid_memory | memory 有效位 | `[B,M]`，bool |
| allowed_memory | 每个 query 能读哪些 memory 项 | 可选 `[B,T,M]`，bool |
| active | 当前轮仍需计算的 token/样本位置 | `[B,T]`，bool |

`True` 在本文所有可见性 mask 中都表示“有效/允许访问”。与其他库对接时必须显式转换，而不是按变量名猜测。

## 4.2 三类状态绝不能混用

**Loop History** 保存同一 token 在历轮的状态：`[E,H1,H2,...]`。它解决“从哪一轮的表示读取”。

**Working Memory** 保存外部证据：检索到的 token hidden 或经过对齐的 Domain SLM slots。它解决“当前有哪些知识可读”。

**Token Self-KV Cache** 保存历史文本 token 在各物理层、各 loop 的 self-attention K/V。它解决“自回归生成时如何复用过去计算”。

因此：

```text
Loop History 的长度：约 1~R，不是 8K。
Working Memory 的长度：例如 1K~8K raw tokens，或 16~64 latent slots。
Token KV 的长度：已处理的文本 token 数，并带 Core 的 loop 轴。
```

本文不改变既有 `memory_budgets` 的总体机制；8K/16K 是资源上限配置，不是 Gate 或 MHAR 的数学常数。不同来源的 raw tokens 与 latent slots 必须分别记录，不能仅按数量声称压缩收益。

# 5. 一轮计算的统一数学定义

设原有共享 Core 有 K 个物理 Transformer blocks，例如 K=3。定义：

\[
E=P_\theta(X),\qquad H^{(0)}=E.
\]

第 r 轮使用的历史为：

\[
\mathcal V_r=\{E,H^{(1)},\ldots,H^{(r-1)}\}.
\]

注意：r=1 时只有一个来源 E；不要同时加入 `anchor` 和同一个 `h0` 造成重复计权。

每轮执行：

\[
\widetilde H_r=\operatorname{LoopMHAR}(\mathcal V_r),
\]

\[
Z_r=H^{(r-1)}+\eta\big(\widetilde H_r-H^{(r-1)}\big),
\]

\[
U_r=\operatorname{CoreWithReader}(Z_r,W_r),
\]

\[
g_r=\operatorname{UpdateGate}\big(H^{(r-1)},E,U_r\big),
\]

\[
\boxed{H^{(r)}=H^{(r-1)}+g_r\odot\big(U_r-H^{(r-1)}\big).}
\]

其中，`η` 是迁移 warm-up 的历史混合强度：从头训练默认 1；从旧权重迁移时可从 0 平滑提升至 1。它不是第四个必须学习的智能控制器。

最后，Halt Router 读取**提交后的 H^(r)**，决定是否进入下一轮；若继续，Retrieval Router 再决定下一轮的 W 是否更新。

**关键禁止项：** `U_r` 已包含 Core 内部 residual，不能写成 `H_next = H_prev + g * U_r`。那会重复叠加旧状态。也不能在更新式的 bypass 中把 `H_prev` 偷换成 `Z_r`，否则 `g=0` 时仍可能通过 MHAR 改写状态。

# 6. Loop-MHAR：按 feature 子空间读取历轮状态

## 6.1 来源与归一化

堆叠历史：

\[
V\in\mathbb R^{N_r\times B\times T\times D},\qquad N_r=r.
\]

对每个来源的完整 D 维做 RMSNorm，仅用于计算 key：

\[
K_j=\operatorname{RMSNorm}(V_j).
\]

再把 D 维拆成 H_hist 个头，每头 `d_h=D/H_hist`。

Value 使用原始的 V，而不是把 RMSNorm 后的 K 当作 Value。否则会把“如何打分”和“实际读回什么状态”混成另一种算法。

## 6.2 每头独立路由

使用一组跨 loop 共享的可训练 pseudo-query：

\[
q^{hist}\in\mathbb R^{H_{hist}\times d_h}.
\]

对来源 j、第 h 个头：

\[
s_{r,j,h}=\langle q^{hist}_h,K_{j,h}\rangle+b_{\operatorname{type}(j,r),h}.
\]

source type 分为 Anchor、较早的 committed state、最新 committed state。类型偏置是本方案的迁移稳定性扩展。

\[
\alpha_{r,j,h}=\operatorname{softmax}_{j}(s_{r,j,h}),
\]

\[
\widetilde H_{r,h}=\sum_j\alpha_{r,j,h}V_{j,h}.
\]

拼接各头恢复 `[B,T,D]`。

**Softmax 只能沿历史来源 j 轴执行，不能沿 token、batch 或 head 轴执行。** 此算子不做跨 token 读取，因此不会自行引入未来 token 依赖；前提是历史状态本身因果正确。

参考实现沿用 pseudo-query 内积，不额外加 `1/sqrt(d_h)`；若改变该缩放，必须同步初始化与实验口径。

## 6.3 为什么采用“整轮状态”而不是“所有子层增量”

这是计算和实现边界的选择。K=3、R=4 时，访问所有子层输出需要更多来源和显存读写；只保存 Anchor 与 committed-loop-state，可以在保持当前 Core 内部 PreNorm 结构的同时，对历轮状态做有限的重新组合。

它也意味着本方案与原版 AttnRes/MHAR 不等价：来源是已经包含历史信息的累积状态，相关性可能很高。多头组合可能帮助恢复有用表示，也可能组合出不一致的子空间。本文不承诺该方法消除信息稀释，更不把它解释为“自动找回正确答案”。

## 6.4 初始化

默认：

```yaml
loop_mhar:
  enabled: true
  heads: 4
  shared_query_across_loops: true
  query_init: zeros
  latest_source_bias_init: 4.0
  other_source_bias_init: 0.0
  key_norm: full_width_rmsnorm
  history_values: committed_states
  detach_history_in_training: false
```

在 query 为零、最新来源偏置为 4 时，若共四个来源，最新来源初始权重为：

\[
\frac{e^4}{e^4+3}\approx0.948.
\]

这不是严格“只读最新状态”。严格兼容测试应把 `η=0`，而不是依赖很大的有限 softmax 偏置。

## 6.5 历史生命周期

训练时，`history.append(committed)`，不在 R≤4 的主训练路径中 `detach()`；否则早期 loop 收不到后续历史读取路径的梯度。

同一 forward 内不能 inplace 修改已经存入 history 的 Tensor。迁移到 fused buffer 时，要重新验证 autograd 与 checkpointing；参考版先用不可变列表与明确 stack。

在自回归 decode 时，历史列表只针对**当前新 token 的历轮状态**建立；过去文本的信息由 self-KV 读取。不应把上一输出 token 的 `H4` 当作下一 token 的 `H0`。

# 7. Gated Recurrent Update：控制本轮候选的写回

## 7.1 默认门控网络

定义：

\[
\Delta U_r=U_r-H^{(r-1)}.
\]

构造：

\[
f_r=\operatorname{Concat}\big(
\operatorname{RMSNorm}(H^{(r-1)}),
\operatorname{RMSNorm}(E),
\operatorname{RMSNorm}(\Delta U_r)
\big).
\]

用低秩 MLP：

\[
a_r=W_2\operatorname{SiLU}(W_1f_r+b_1)+b_2,
\qquad g_r=\sigma(a_r).
\]

其中 `W1: 3D→k_g`，`W2: k_g→D`，默认 `k_g=64`。门为 `[B,T,D]`，逐 token、逐通道控制写回。

最终：

\[
H^{(r)}=(1-g_r)\odot H^{(r-1)}+g_r\odot U_r.
\]

- `g=0`：保留旧状态。
- `g=1`：完全采用候选。
- 中间值：逐通道插值。

这只是向量层面的状态更新。不能声称某个坐标天然对应“数学知识”或“语法”；这类含义最多是解释性假设。

## 7.2 Anchor 的作用边界

Anchor E 在每轮 Gate 中可见，也始终是 MHAR 的候选来源。它提供一个固定参照，有助于模型学习何时保留原始输入表示。

但“可读取 Anchor”不意味着模型一定使用它，也不是严格防止语义漂移的约束。应记录 Anchor 权重和任务结果，而不是强行要求 Anchor 的权重始终大于某个阈值。

## 7.3 数值与稳定性边界

插值形式可限制**每个坐标**落在旧值与候选值之间，但不保证完整向量范数下降。

例如 `h=(1,0)`，`u=(0,1)`，`g=(0,1)`，则输出为 `(1,1)`，其 L2 范数比 h、u 都大。

而且 Gate 本身依赖状态。简化地把其他来源固定，有：

\[
\frac{\partial H'}{\partial H}
=I+\operatorname{diag}(g)(J_U-I)
+\operatorname{diag}(U-H)J_g.
\]

所以 `0≤g≤1` 并不自动给出整个循环的 contraction 或梯度稳定保证。仍需 FP32 门控计算、梯度裁剪、范数监控和长 loop 验收。

## 7.4 初始化与梯度

默认 `W2=0`，`b2=+4`：

\[
g\approx\sigma(4)=0.9820.
\]

这样初始行为接近旧的“接受 Core 输出”，但不是严格相等。与作者 GRT 模型卡中近开门初始化的思路一致；具体低秩结构是本方案定义。[R3]

该初始化下，第一步 `W2/b2` 可有梯度，而 `W1` 可能因 `W2=0` 暂时没有梯度。这是初始化的正常结果，不应要求所有层第一步梯度都非零；应在后续若干步检查整条路径。

默认不加 state noise、gate noise，也不加让 `mean(g)` 持续减小的正则。

## 7.5 Update Gate 与 Halt Router 的分工

Update Gate 决定“算出来的候选写入多少”；Halt Router 决定“下一轮是否再执行”。

Halt 可读取以下统计作为额外特征：

\[
\overline g_r,\quad
\Delta^{candidate}_r=\frac{\operatorname{RMS}(U_r-H^{(r-1)})}{\operatorname{RMS}(H^{(r-1)})+\epsilon},
\]

\[
\Delta^{commit}_r=\frac{\operatorname{RMS}(H^{(r)}-H^{(r-1)})}{\operatorname{RMS}(H^{(r-1)})+\epsilon}.
\]

推荐：这些数值特征先 `detach()` 再喂 Halt，以免 Halt 辅助损失通过统计量诱导 Update Gate 刻意关闭。

**不采用此前脑暴中的硬绑定** `P(continue)≈mean(g)`；也不以 `Δ_commit` 很小直接生成“正确停止”的标签。它们是状态变化特征，不是正确性证据。

# 8. Memory Reader Gate：独立读取和有界注入外部证据

## 8.1 插入位置必须唯一

默认只在 Core 的第一个物理 block 内：

```text
输入 Z_r
  ↓
Self-Attention + 原有 residual
  ↓
Memory Cross-Attention + Reader Gate + 独立 residual
  ↓
FFN + 原有 residual
  ↓
Core 的其他物理 block
```

K=3、R=4 时，最多执行四次 memory read，而不是十二次。检索次数仍可以只有一次；“从已缓存 W 读取”与“重新搜索全库”是两笔不同成本。

若以后把 Reader 扩到多个 Core 层，必须在配置、FLOPs、参数量和 KV 缓存中显式体现。首版不同时做 pre-core Reader 与 per-layer Reader，避免重复注入。

## 8.2 数学定义

第一个 block 的 self-attention 之后：

\[
A_r=Z_r+SA_0(N_0(Z_r)).
\]

独立 memory 分支：

\[
Q=W_Q^mN_q(A_r),\qquad
K=W_K^mN_m(W_r),\qquad V=W_V^mN_m(W_r).
\]

\[
m_r=W_O^m\operatorname{Attention}(Q,K,V;\text{visibility mask}).
\]

token 级 Reader Gate：

\[
c_r=\sigma\Big(W_{m,2}\operatorname{SiLU}(W_{m,1}[N_q(A_r);N_o(m_r)])+b_m\Big),
\]

\[
\rho=\tanh(a_m),\qquad
\overline A_r=A_r+\rho\,c_r\odot m_r.
\]

随后执行 FFN 及其余 Core 层，得到 U_r。

默认 `c_r:[B,T,1]`，可广播到 D 维。Reader Gate 和 Update Gate 不需要相同粒度：前者先控制整条外部证据支路，后者逐通道提交候选。

`ρ` 是可训练、有界的有符号残差缩放，不是置信度概率；`c_r` 也不应未经校准就向用户显示为“知识可信度”。

## 8.3 不再混合两套 K/V 坐标系

新 Reader 的 Q、K 都不使用 token self-attention 的 RoPE。原 self-attention 保留自己的正常位置编码。

基线将已 RoPE 的 self-query 与未 RoPE 的 memory key 放在同一 softmax 内。新方案通过独立 reader 投影和注意力消除这种隐式混用。[C2]

memory 内的顺序、文件来源和版本信息，要由 Memory Encoder/Bridge 及 metadata 明确提供；独立 Cross-Attention 自己不能恢复已经被 chunking 丢掉的全局位置信息。

## 8.4 空记忆与 mask 的严格语义

- `M=0` 或 `memory=None`：Reader 直接返回输入，保证 no-op。
- 某个 batch 行没有有效 memory：该行必须 no-op，不产生 NaN。
- 某个 query 由于可见性约束不能读任何真实 memory：同样 no-op。
- padded memory 不得进入 norm、score 或统计量。
- 训练/推理均支持 `allowed_memory[B,T,M]`；它控制因果/可用性，而不是假设整个 batch 共享同一份可见内容。

参考实现增加仅在“无真实可见项”时启用的零值 NULL K/V，以规避全遮罩 softmax 的问题；随后把该 query 的注入 delta 严格置零。NULL 不作为无条件可选知识源。

## 8.5 初始化不要“双零锁死”

从头训练默认 `ρ≈0.01`，Reader Gate 初始约 0.5，`W_O^m` 正常随机初始化。

迁移旧模型、需要初始精确 no-op 时，可令 `a_m=0 → ρ=0`，但 `W_O^m` 必须保持非零初始化。第一步先更新缩放，之后再使 Reader 内部参数获得任务梯度。

不能同时使用：

```text
reader scale = 0
且
reader output projection = 0
```

否则两者可能互相使对方梯度为零，导致支路无法启动。

## 8.6 Reader Gate 不替代证据验证

外部 memory 可能错误或不适用。Reader Gate 提供降低注入的手段，但不是事实判别器的保证。

数据必须包含相关、无关、相似但版本不适用、冲突和缺失证据；评测必须检查这些情况，而不能仅检查“加 memory 后 logits 有变化”。

# 9. 与 Dynamic Retrieval / Adaptive Working Memory 的执行协议

## 9.1 每轮动作顺序

必须固定为：

```text
1. 读取上一轮 committed state 与当前 Memory 版本
2. Loop-MHAR
3. CoreWithReader 生成候选
4. Update Gate 提交新状态
5. 计算 Halt logits
6. 若停止，直接进入 Coda；若已到 R_max，也禁止无用的再检索
7. 若继续，计算 Retrieval / Budget logits
8. 按动作更新 Memory，增加 memory_version
9. 开始下一轮
```

不能先昂贵地检索，再发现已经停止；也不能先用候选 U 决定检索，随后 Gate 几乎拒绝 U，却仍按被拒绝的状态执行动作。

## 9.2 三种 Retrieval 动作

默认动作空间：

\[
a^{ret}\in\{\text{REUSE},\text{RESCORE\_WARM},\text{GLOBAL\_REFRESH}\}.
\]

- `REUSE`：沿用当前 Hot Memory，不重建索引，不重投影 K/V。
- `RESCORE_WARM`：在已缓存候选中重新选择；可能改变 memory_version。
- `GLOBAL_REFRESH`：基于当前可见状态重新搜索全局索引，再更新候选与 Hot Memory。

如果现有实现还没有 Warm Candidate Cache，`RESCORE_WARM` 必须被标成不可用并在 logits 中 mask 掉，不能把它悄悄实现为全局搜索后仍按低成本计费。

## 9.3 Memory 更新协议

Memory 必须保留 `source_id/block_id`，并按这些 ID 去重。重复检索同一块，应该更新 usage/recency，不应把同一知识复制成多个 slot，虚增 memory_size 和 attention 权重。

建议版本字段：

```python
@dataclass
class WorkingMemoryV2:
    hidden: Tensor                 # [B,M,Dm]
    valid: Tensor                  # [B,M]
    source_ids: Tensor             # [B,M] 或 token→block 映射
    source_versions: list[str]
    memory_version: int
    metadata: dict
```

原始 block memory 可先按 `[B,K,block,D]` 管理，再展平给 Reader。Domain SLM slots 通过同一接口进入，但 metadata 必须说明来源类型与 Bridge 版本。

预算 Router 决定的是合法容量上限；如果内容不足，实际使用可以更少。日志必须分别记录 `budget_cap_tokens`、`actual_valid_tokens`、`unique_blocks`，不能用分配容量冒充有效证据量。

## 9.4 新证据不应被频繁关闭的 Gate 永久挡住

当 memory_version 改变时，可把 `memory_changed`、新增有效块数量、检索分数差等作为 Halt/控制器输入，并允许其决定多算一轮。

但不采用“只要检索就强制 Gate=1”。因为新证据也可能无关或错误。新增证据样本应进入训练，验证模型可以重新打开 Reader/Update Gate，而不是靠硬覆盖旧状态实现。

# 10. 因果性与训练/推理一致性

## 10.1 数据字段必须分开

训练样本至少区分：

```json
{
  "source_context": "外部可查询资料",
  "prompt": "当前用户已经提供的完整问题",
  "completion": "监督答案",
  "evidence_spans": [],
  "source_version": "...",
  "task_id": "..."
}
```

不能只拿 `input_ids = prompt + completion` 再通过全序列平均池化构造检索 query。当前附件的 SFT-C 有这条风险，必须在接入新模块前修复。[C4–C6]

## 10.2 初始检索与后续检索的可见边界

初始检索可以读取**完整已知 prompt**，因为用户已提供全部问题；但不能读取 completion 的任何 gold token。

生成第 t 个答案 token 时，控制器只允许使用：

\[
\{\text{prompt},y_{<t},\text{当前可见 Memory},\text{截至当前的 loop states}\}.
\]

teacher forcing 中已经提供的 `y_<t` 可用于该位置的状态计算，但 `y_t` 与更后的 gold tokens 只能用于计算 loss，不能参与此位置的检索或门控决策。

纯语言模型预训练没有完整“用户问题”边界时，不能借口 question-conditioned memory，把整条训练序列作为检索 query；每个预测位置必须遵守实际可见前缀。

## 10.3 正确的验收反例

固定相同 prompt 和 source_context，替换完整 gold completion：**初始检索结果必须不变**。

固定相同答案前缀 `y_<t`，替换 `y_t:`：**第 t 个位置的 memory action、Halt logits、Reader 输出与 next-token logits 必须不变**，除数值容差外。

这里比一般的 attention causal mask 更严格，因为数据也可能通过 Router、预算、全序列池化或 memory 选择泄漏。

## 10.4 动态检索不能假装普通并行 SFT

同一个 `[B,T,D]` 的全序列平均 query，无法同时代表 T 个不同生成前缀的检索需求。

终局训练必须提供两种执行器，并保证语义对应：

1. **逐前缀/逐 token 的参考执行器**：完整保留动态 Memory 与动作轨迹，作为正确性标准。
2. **按 memory-version 与动作边界分段的执行器**：一个 segment 内的 Memory 在 segment 开始前已经确定，段内可以批处理；需要中途 refresh 时先结束 segment，再决策和继续。

训练使用事先给定的动作边界，也必须确保实际检索 query 仅由当前可见前缀计算。不能先用答案全序列算出“最相关的 Memory”，再把它广播给更早位置。

这属于同一终局模型的参考路径与优化路径，不是两套竞争架构。

# 11. KV、History 与缓存失效契约

## 11.1 三类缓存的生命周期

| 缓存 | 内容 | 生命周期 | 允许跨 optimizer step？ |
|---|---|---|---|
| Loop History | 当前 token/chunk 的历轮 committed states | 当前前向内 | 否 |
| Memory KV | Reader 专属投影后的外部 K/V | 同一 memory/projection 版本 | 训练默认否；推理权重不变时可以 |
| Token Self-KV | 各层、各轮历史文本的 K/V | 当前生成请求 | 训练依执行器；推理按请求保持 |

训练时 Memory KV 可在同一计算图中跨 R 个 loop 复用，梯度会汇总。不要把它 `detach()`，也不要跨多次独立 backward 复用旧图。

推理时建议 cache key 至少包含：

```text
request_id
memory_version
reader_layer_id
projection_weight_version
payload_encoder_or_bridge_version
device
dtype
```

若训练 Memory Encoder，也要包含 encoder 更新版本。仅用“同一个 WorkingMemory Python 对象”判断缓存是否有效不够。

## 11.2 固定 R 下的 self-KV

Core self-KV 按物理层与 loop 维分开：

```text
core_kv[layer_id][loop_id] = token-prefix K/V
```

第 r 轮的当前 query 读取历史 token 的第 r 轮 K/V。不能把上一轮的全部 KV 当作新的文本继续追加；token 轴每处理一个新 token 只增长 1。

每个物理层的 K/V 仍从其真实 self-attention 输入产生，不从 loop 末尾的 H^(r) 重新替换。Update Gate 只改变**进入下一轮/进入 Coda**的状态，不追溯改写已经生成的本轮内部 K/V。

## 11.3 可变 R 的难点：不能丢掉历史深轮次

若某个新 token 在第 1 轮就停止，不能把整个请求的 Core cache 从 R 个槽裁剪成 1 个槽。前面 token 已经算出的深轮次状态仍要保留。

本版本对 adaptive decode 定义明确的 **last-valid-per-token 填充语义**：

- 为每个 Core 层保留 `R_max` 个逻辑槽。
- 新 token 实际执行到 r_t，则其 r≤r_t 的 K/V 使用真实计算值。
- 新 token 的 r>r_t 槽，复制该 token 在该物理层最后实际计算轮次的 K/V。
- **只填充这个新 token 的位置**；不能用浅层整段缓存覆盖先前 token 的深轮次缓存。
- 多样本 batch 中按样本/位置操作，不按 batch 平均深度决定。

示意：

```python
for layer in core_layers:
    for r in range(R_max):
        current_token_kv = actual_kv[layer][min(r, last_executed_round)]
        cache[layer][r] = append_one_token(cache[layer][r], current_token_kv)
```

这只是语义伪代码；batch 版本须使用每样本的 last_executed_round。

**重要：** last-valid 填充是模型定义的近似缺失状态规则，不等价于真的执行全部深层。不能声称它在任意 adaptive-R 下与普通全深度 prefill 严格等价。训练的可变深度阶段要使用相同规则。

## 11.4 Prefill 的明确策略

首版优化路径推荐 `prefill_policy=fixed_cap`：对完整已知 prompt 使用当前模式允许的固定 R_prefill，建立一致 cache；decode 再按 token 自适应停止。相应的 TTFT 与 decode 成本分开记录。

这不改变动态深度模型，只明确 prefill 的批处理策略。若要连 prompt prefill 也逐 token 早停，需要实现与上节相同的 per-token cache 填充及 active-mask 语义，不能仅复用一个 batch-level halt 决策。

## 11.5 动态 Memory 下的 cache 语义

本版本采用 **append-only past**：memory 更新只影响此后执行的计算，过去 token 的 self-KV 不回溯重算。

因此，同一请求在 W1→W2 后的 stateful decode，不应与“拿最终 W2 对整个前缀从头重算”强行比较相等。正确的对照是：按同一 Memory 版本时间线、同一动作序列进行参考重放。

必须保存：

```text
(token_position, loop_index, memory_version_used,
 retrieval_action, budget_action, exit_round)
```

这份轨迹也是后续策略训练的必要输入。

## 11.6 Checkpointing 必须是纯计算重放

梯度 checkpoint 的函数内部不要执行全局检索、修改 MemoryManager、更新统计计数或改变 memory_version。反向重计算可能再次执行 forward，导致重复取数和副作用。

正确做法：控制器先得到不可变 memory payload/KV，checkpoint 只接收 Tensor、mask 与确定的配置，重算纯 Reader/Core 运算。

# 12. PyTorch 模块和模型接口设计

## 12.1 新增与修改文件

```text
model/minimind_x/
  residual/
    __init__.py
    loop_mhar.py                 # 新增
    recurrent_update.py          # 新增
  memory/
    reader.py                    # 新增：独立 Cross-Attention + gate
    working_memory.py            # 增加 ID、valid mask、版本信息
    memory_manager.py            # 去重、有效容量、版本更新
  router/
    halt_router.py               # 分离“算概率”和“执行停止”
    retrieval_router.py          # 可用动作 mask 与轨迹
  thinking_core.py               # 核心集成点
  minimind_x.py                  # 显式输出 committed/core states
  controller.py                  # 因果 query、grad 边界、Memory 版本
  losses.py                      # 真实接入 Router 与成本损失
  flops.py                       # 分项计数，不使用容量代理真实开销

trainer/
  residual_schedule.py          # 新增：同一训练 run 的 warm-up
  oracle_labels.py               # 同前缀反事实，不读取 gold 后缀
  train_pretrain_x.py
  train_sft_x.py
  train_grpo_x.py                # 如启用策略 RL，必须能重放动作

tests/
  test_residual_invariants.py
  test_reader_gradients.py
  test_routing_causality.py
  test_dynamic_cache_replay.py
```

## 12.2 核心模块 API

```python
class LoopMHAR(nn.Module):
    def forward(self, history: list[Tensor], strength: float = 1.0):
        # history: 每项 [B,T,D]
        # returns: mixed [B,T,D], weights [N,B,T,H_hist]
        ...

class RecurrentUpdateGate(nn.Module):
    def forward(self, previous, anchor, candidate,
                active=None, force_gate=None):
        # returns: committed [B,T,D], gate [B,T,D]
        ...

class GatedMemoryReader(nn.Module):
    def project_memory(self, memory_hidden, valid):
        # TRAIN 时必须在 grad mode 中调用
        # returns MemoryKV: [B,Hkv,M,head_dim]
        ...

    def forward(self, hidden, memory_kv,
                query_valid=None, allowed=None):
        # returns: updated_hidden, gate/delta diagnostics
        ...
```

接口的可运行参考实现见附录 A 及配套参考包。

## 12.3 模型输出新增字段

不要再用同一个 `exit_hiddens` 同时指代“Core 提交状态”和“过 Coda 后的状态”。

```python
@dataclass
class MiniMindXOutputV2:
    logits: Tensor | None
    exit_logits: list[Tensor] | None
    core_committed_states: list[Tensor] | None
    coda_exit_states: list[Tensor] | None
    halt_logits: list[Tensor] | None
    retrieval_logits: list[Tensor] | None
    budget_logits: list[Tensor] | None
    exit_round: Tensor | None
    memory_versions: list[int] | None
    router_trace: object | None
    residual_stats: dict
    past_key_values: object | None
```

生产推理默认不返回完整 history 与 attention weights，避免日志保留计算图或泄漏原始知识。调试时显式开启。

## 12.4 Core 集成伪代码

以下省略具体 self-KV 拼接，仅展示精确顺序；`CorePass` 内首层已经包含一次 Reader。

```python
anchor = prelude_hidden
previous = anchor
history = [anchor]

for r in range(max_loops):
    read_state, hist_weights = self.loop_mhar(history, strength=history_strength)

    # 记忆由外部 controller 按确定版本提供；这里不执行全局检索。
    candidate, current_pass_kv, reader_stats = self.core_pass(
        read_state,
        loop_index=r,
        token_past_kv=token_past_kv,
        memory_kv=memory_kv,
        token_mask=token_mask,
        memory_visibility=memory_visibility,
    )

    gated, update_gate = self.recurrent_update(previous, anchor, candidate,
                                               active=active)
    # 仅迁移 warm-up 使用；从头训练 update_progress=1。
    committed = candidate + update_progress * (gated - candidate)
    committed = where_active(active, committed, previous)

    stats = build_gate_features(previous, candidate, committed, update_gate)
    # features 中用于控制器的统计量 detach；h/e 是否 detach 由训练阶段决定。
    halt_logits = self.halt_router(committed, anchor, stats.detached())
    record_core_exit(committed, halt_logits, hist_weights, reader_stats)
    history.append(committed)
    previous = committed

    if training_full_unroll:
        # 继续用于多出口/动作监督；不把 probability evaluation 与 break 绑在一起。
        action = teacher_or_replayed_action(r)
    else:
        action = controller.decide_after_commit(committed, halt_logits, r)

    commit_current_token_kv(current_pass_kv, active, r)
    if action.stop or r == max_loops - 1:
        break

    memory_kv, memory_visibility = controller.prepare_next_memory(action)

# 使用 selected committed state 过 Coda，而不是未提交候选。
output = coda(select_committed_exit(...))
```

训练全展开模式下，“某个样本监督出口之后的动作是否还计算”必须有对应 mask：可以为辅助出口继续计算，但不能把这些额外动作计入该样本实际策略成本。

# 13. 梯度接线：防止“模块在运行，参数没学习”

## 13.1 默认梯度策略

| 组件 | 从头训练 | 旧模型迁移 warm-up | 备注 |
|---|---|---|---|
| Backbone/Core | 训练 | 可先冻结参数，再逐步解冻 | 冻结参数不等于对前向套 `no_grad` |
| Loop-MHAR | 训练 | 训练，但 η=0 时梯度暂被关掉 | η 要按计划升起 |
| Update Gate | 训练 | 训练/暖启动 | 最后一层零初始化会延迟前一层的梯度 |
| Reader Q/K/V/O | 训练 | 训练 | 不与离散检索一起 `no_grad` |
| Reader Gate/scale | 训练 | 训练 | zero-scale 时先允许 scale 获得梯度 |
| Memory payload encoder | 可训练或共享编码器 | 默认暂冻 | 是否冻结须入配置 |
| Top-K/ID 选择 | 离散操作 | 离散操作 | 不能期待索引本身自动可导 |
| Retriever query/key 投影 | 用显式检索损失训练 | 可训练 | 必须使用带图的 scores |
| Halt/Retrieval/Budget Router | 课程中训练 | 课程中训练 | logits 必须真正参与 loss |

PyTorch 的 `no_grad` 会阻断其中操作的计算图，即使参数 `requires_grad=True`。因此现有把 Memory 编码、选择、KV 投影一起放入 `no_grad` 的做法，不能用于声称 Reader 投影已从答案 loss 学习。[R5；C3]

## 13.2 推荐的内存分支写法

```python
# 1. 离散选择可以在 no_grad 下执行。
with torch.no_grad():
    selected_ids = retriever.select_ids(causal_query, index)
    frozen_payload = frozen_memory_encoder(selected_ids)

# 2. Reader 投影保留梯度，同一 forward 内跨 loops 复用。
mem_kv = memory_reader.project_memory(frozen_payload, valid_memory)

# 3. Core 即使冻结参数，只要输入依赖待训练模块，也不能整体 no_grad。
outputs = core_and_coda(..., memory_kv=mem_kv)
answer_loss = loss_fn(outputs, targets)

# 4. 独立重算带图的检索分数。
retrieval_logits = retriever.trainable_scores(causal_query, candidate_features)
retrieval_loss = retrieval_supervision(retrieval_logits, gold_ids)

(answer_loss + weight * retrieval_loss).backward()
```

如果希望答案 loss 直接训练“选哪一块”的 query 投影，必须另外设计软加权、可微选择近似或离散动作策略梯度。仅取消 `no_grad` 后仍使用硬 Top-K indices，并不能自动产生这条梯度路径。

## 13.3 梯度验收不能只看整体 Core moved

必须分别检查：

```text
loop_mhar.query
loop_mhar.type_bias
recurrent_update.fc2
recurrent_update.fc1（若最后层零初始化，检查后续步）
memory_reader.q_proj / k_proj / v_proj / o_proj
memory_reader.scale_logit
memory_reader.gate_fc2 / gate_fc1
halt_router
retrieval_router
budget_router
retriever.query_proj / key_proj
```

区分三种情况：应训练但无梯度、初始化允许暂时无梯度、明确冻结。不同情况不能混在一张“梯度通过”的总表中。

本文配套测试验证前三个新增模块的独立梯度；原仓库真实 trainer 与 Router 接线仍需按本节验收。

# 14. 初始化、权重迁移与同一训练 run 的课程

## 14.1 两种入口，一套最终架构

### 从头训练

```text
history_strength η = 1
update_progress = 1
history latest bias = +4
Update Gate 最后层 weight = 0，bias = +4
Reader scale ρ ≈ 0.01
Reader 内容门初值 ≈ 0.5
```

所有新模块从一开始就在模型图里。循环数按既定 depth-elastic 训练计划采样，初始不让随机 Halt 决策控制实际路径。

### 从既有 MiniMind-X 权重迁移

需要两个不同概念：

**严格兼容模式：** MHAR bypass；Update 强制采用 U；无 memory 或 Reader scale=0。用于与旧模型的无 memory 路径做数值回归，不作为长期训练配置。

**可学习迁移模式：** history_strength 与 update_progress 在同一 job 中逐步升起；Reader 保持非零输出投影，让零 scale 可被优化器打开。

## 14.2 建议迁移课程

以下比例是工程起始值，不是论文结论，也不是训练完成时间预估。

| 优化进度 | history_strength | update_progress | 训练内容 |
|---|---:|---:|---|
| 初始化检查 | 0 | 0 | 无 memory 旧模型 logits 回归；不更新 |
| 前 1% optimizer steps | 0→0.25 | 0→1 | 新模块启动，先保持接近旧路径 |
| 1%–5% | 0.25→1 | 1 | 三个模块完全接入；不要因 Gate 小而提前停 |
| 后续主训练 | 1 | 1 | 任务/多深度/有无 memory 混合训练 |
| 策略收尾 | 1 | 1 | Router 监督、适度成本约束、on-policy 校准 |

迁移 warm-up 的更新式可写为：

\[
H^{(r)}_{warm}=U_r+\lambda_{warm}(H^{(r)}_{gated}-U_r).
\]

当 λ_warm=0 时严格为 U；当 λ_warm=1 时为最终 Gate 更新。训练日志同时记录 raw gate 与 effective gate，避免 warm-up 时错误解读。

## 14.3 旧 Memory 拼接路径不能假装无损转换

对于旧的 `cat(memory_KV, token_KV)` 联合 softmax，拆成独立 cross-attention 后，即使权重维度相同，概率归一化也改变了。

因此：

- 可以保留原 self-attention 的主干权重。
- Reader Q/K/V/O 可以按尺寸兼容情况初始化，但必须记录来源。
- 无 memory 的兼容性可以严格测试。
- **有 memory 的旧新输出不保证初始相等**，需要任务训练或旧模型行为蒸馏。

迁移工具必须输出新增、缺失、尺寸变化与未加载的键清单；`strict=False` 不是忽略配置冲突的许可证。

## 14.4 训练超参数建议

保留已经验证的 backbone optimizer/学习率方案，新模块采用 1×–2× backbone LR 作为起点，norm/bias/gate scale 不施加普通 weight decay。

不在本增量文档中重新规定总 token 量或宣称若干天能完成训练。实际成本取决于模型尺寸、Memory 长度、每轮 Reader 开销、oracle probe 比例与硬件 kernel。

# 15. 损失函数与自适应控制器监督

## 15.1 三个新模块主要从任务 loss 学习

默认不需要人工给 Update Gate 标注 0.2/0.8，也不需要人工标注“第 2 个 history head 应读取第 1 轮”。

对固定/随机深度和多出口任务 loss：

\[
L_{answer}=\sum_{r=1}^{R}w_r\,CE(\operatorname{Head}(\operatorname{Coda}(H^{(r)})),y).
\]

只在合法监督位置计算 CE。不同出口都经过同一套 Coda，Coda 额外前向成本要计入训练 FLOPs。

Gate、MHAR 与 Reader 从答案误差反传中学习如何使用状态与证据。

## 15.2 推荐总目标

\[
L=L_{answer}
+\alpha_hL_{halt}
+\alpha_rL_{ret}
+\alpha_bL_{budget}
+\alpha_kL_{retriever}
+\alpha_mL_{reader\_aux}
+\lambda_C\mathbb E[C]
+\alpha_{KD}L_{KD}.
\]

推荐初始接线：

| 项 | 起始设置 | 说明 |
|---|---:|---|
| L_answer | 1.0 | 主目标 |
| L_halt | 0.1 | 只在可信 oracle 决策位置监督 |
| L_ret | 0.1 | 复用/重排/全局检索的动作监督 |
| L_budget | 0.1 | 真正改变可见 Memory 的预算对照 |
| L_retriever | 0.1 | gold span/block 或任务效用监督 |
| L_reader_aux | 默认 0 | 有明确正负适用性标签后再开 |
| E[C] | 初始 0，随后从小量升起 | 先让路径有效，后优化成本 |
| L_KD | 默认 0，可按迁移需要开启 | 教师与学生对齐合法输入和任务位置 |

这些权重是项目建议，不是已经在 MiniMind-X 上验证的最优超参数。

## 15.3 不采用的“看起来漂亮但容易误导”的损失

不默认使用：

\[
|P(continue)-\operatorname{mean}(g)|,
\]

也不使用 `λ * mean(update_gate)` 冒充算力惩罚。

原因：前者把状态改变量误当成正确性/剩余推理量；后者可能奖励关闭 Gate，但 Core 已经算过，计算并没有省掉。

也不强迫每个 loop 的 Gate 均值递减。某个后续 loop 新增关键证据时，合理行为可能是重新大幅更新。

## 15.4 Halt Oracle 的定义

目标是“在合法可见前缀与给定 Memory 状态下，最省计算且任务足够好的停止点”。

对完全答不对的样本，标记 `unresolved`。不能机械地把它们都叫作“需要最大思考深度”。可以使用 loss 饱和信号作为弱监督，但必须与成功 oracle 分开统计。

teacher-forced CE 接近也不等于自由生成成功率相同。用于部署的 Halt 阈值，需要用真实生成/执行验证做校准。

## 15.5 Retrieval Oracle 必须是“是否再检索”的反事实

在同一决策状态 s、同一 W 上比较：

\[
Q(\text{REUSE}\mid s,W)
\quad\text{与}\quad
Q(\text{REFRESH}\mid s,W\rightarrow W').
\]

不能用“完全无 memory”与“已有 memory”的差来替代；后者只说明 memory 有没有用，不说明当前还需不需要刷新。

同一 probe 应固定问题、前缀、checkpoint、随机条件和后续计算预算。真正找回新证据后，才有资格评估 refresh 的边际收益。

## 15.6 Budget Oracle 必须保证档位真的改变计算路径

对预算 B 执行：

```text
计算目标有效容量
→ 得到可选候选
→ 按容量选择不同的有效 token/block
→ 生成对应 Reader KV
→ 以同一深度与相同条件评估 loss/正确性
```

若各档实际都是固定 `topk_fine=8` 块、没有有效内容差异，那么 sweep 得到的是退化对照，不是 Memory Budget 的学习信号。

## 15.7 成本估计须尊重条件依赖

停止分布可由 `p_stop` 计算：

\[
\pi_r=p_r\prod_{j<r}(1-p_j),
\]

最后一轮强制吸收剩余概率。

但当 Memory Budget、检索与深度互相依赖时，不应粗暴使用 `E[R] * E[M]` 代替真实联合成本。应使用实际状态条件下的期望或采样轨迹的累计成本：

\[
C(\tau)=C_{pre}+C_{coda}
+\sum_{r\in\tau}\big(C_{core,r}+C_{hist,r}+C_{reader,r}+C_{gate,r}\big)
+\sum_{a\in\tau}C_{retrieval,a}.
\]

只对动态控制器的 action probabilities 或动作 reward 建立成本梯度。实际整数计数继续用于日志和性能报告。

# 16. 与本次模块相匹配的数据设计

## 16.1 保留原有能力数据，新增“对照式证据条件”

不需要把所有 SFT 都改成很长的显式思维链。优先新增如下样本族：

| 样本族 | 关键构造 | 训练/验收目标 |
|---|---|---|
| 无外部证据 | memory=None | Reader 不应扰动无 memory 路径 |
| 外部证据必需 | 答案含本批次新生成的随机事实，prompt 无法推断 | 证明 Reader 真的使用了外部信息 |
| 无关 memory | 与问题无关、长度/格式相似的资料 | 不因 memory 存在就盲目注入 |
| 相似但不适用 | 相同术语，不同版本、对象或生效条件 | 学习适用性，而非只匹配关键词 |
| 新证据纠正旧假设 | 后续 loop 才检索到关键约束 | 验证 Gate 可以重新打开，不锁死旧状态 |
| 多跳证据 | 后续检索 query 依赖前一次读取的信息 | 验证 Think→Retrieve→Think 的闭环 |
| 证据冲突/不足 | 多来源冲突，或确实缺少必要字段 | 允许继续检索或承认不足，而非强制回答 |

## 16.2 防止可学习性测试走捷径

只在固定输入上过拟合，或者把标签构造成直接复制当前输入 token，不能证明 memory 或深 loop 提供了收益。

应加入“同一问题模板、每次随机改外部事实”的任务，例如：

```text
Memory: 本次随机配置中，设备 A 的阈值为 7319。
Prompt: 本次设备 A 的阈值是多少？
Target: 7319
```

下一条样本把阈值换成另一随机值，训练与验证使用不重叠的事实组合。可通过移除/交换 memory 检验答案是否正确随证据变化。

这个任务只是机制验收；真实领域能力仍需真实文档、代码和版本约束任务。

## 16.3 Gate 辅助监督只使用清晰标签

Reader 的“无关/版本不适用”样本可以对分支使用施加弱约束，但不能把所有错误答案都归因于 Reader Gate，也不能把高 attention weight 当作因果贡献标签。

困难样本有可能需要 memory，但 backbone 尚不能正确利用。把这种样本全部训练成“关 Reader”会使系统逃避知识读取。

## 16.4 数据分割与验证

按文档来源、仓库、任务模板或事实生成种子划分 train/validation/test，避免同一事实换一个 prompt 进入测试集。

gold evidence 必须基于**真实 tokenizer 之后的 token span/block ID**，不能用字符比例估算后直接拿来训练精确检索 CE。tokenizer/分块参数改变时，重新生成标注。

# 17. RL 与计算策略学习的边界

## 17.1 固定 R 的 token GRPO 能做什么

固定 R rollout，并在相同 R 下重算 token log-prob，可以训练 backbone 与可微的 Gate/Reader/MHAR 路径。但如果训练没有 memory 输入，这次 rollout 自然不能优化 memory 使用。

更重要的是：只计算输出 token 的 log-prob，不能自动训练未纳入动作概率的 Halt、Retrieval、Budget 决策。不要因为 reward 中写了 memory/loop cost，就宣称这些策略已被优化。

## 17.2 真正的 Adaptive Policy RL 要记录什么

对每条轨迹保存：

```text
prompt/source IDs
每个决策点的可见前缀
每个 token/loop 的 exit 与 active mask
retrieval action 与 selected evidence IDs
budget action
memory_version
动作 log-prob
输出 token log-prob
温度、采样过滤规则与 checkpoint 版本
```

联合路径概率包含：

\[
\log P_\theta(\tau)
=\sum_t\log P_\theta(y_t\mid s_t)
+\sum_i\log\pi^h_\theta(a_i^h\mid s_i)
+\sum_i\log\pi^r_\theta(a_i^r\mid s_i)
+\sum_i\log\pi^b_\theta(a_i^b\mid s_i).
\]

Update Gate 是连续可微模块，不需要另外把每个通道 gate 当成离散 RL action；它可通过任务/策略损失的计算图获得梯度。

## 17.3 禁止平均深度重放

生成时如果各 token 深度为 `[1,3,2,4]`，不能把平均值取整为 3 后，用所有 token 都 R=3 的前向估计原轨迹概率。

同理，不能用最终 memory 替换轨迹中每一轮的真实 memory。重放协议必须与第 11 节完全一致。

## 17.4 Reward 的最低约束

正确性/任务成功是主项，效率是次项。成本范围归一化并设置上界，避免正确但稍贵的轨迹被评分低于错误但便宜的轨迹。

代码执行 reward 必须运行在隔离沙箱中：无训练目录写权限、默认无网络、受限 CPU/内存/进程数并有超时。`subprocess.run(python -c ...)` 本身不是沙箱。

主线先通过 SFT/任务监督训练三个新模块；完整 Adaptive Policy RL 只在重放一致性测试通过后启用，不把它设为残差改造的隐性必需依赖。

# 18. 算力、显存与真实性能预算

## 18.1 Gate 本身不是跳计算

本方案 Update Gate 依赖 U_r，因此每次关 Gate 前，Core 已经完成计算。

报告时分别统计：

```text
executed_core_rounds
selected_exit_rounds
active_token_rounds
physical_batch_rounds
```

一个 batch 为等待最慢样本而继续执行，不能只用样本平均停止轮次宣称相同比例的 GPU 加速。压紧 active 样本后才可能兑现部分计算节省。

## 18.2 新增参数估算

令 D=1024、update rank=64：

- Update Gate 两个线性层主要参数约 `4*D*64 = 262,144`，另有 bias/norm。
- Loop-MHAR 的共享 query 共 D 个参数；另有 norm 与少量 source-type bias，不因四头而扩大到 4D query。
- Reader 若 `query_heads=8, kv_heads=2, head_dim=64, D_m=1024`，Q/K/V/O 主要参数约 1,310,720，另有 gate/norm。

这些是本文配置的解析算式，不是现有整个 MiniMind-X 的实测参数量。最终 `sum(p.numel())` 要输出到 manifest。

## 18.3 Loop History 显存

保存 R+1 个状态的张量存储上界近似为：

\[
(R+1)BTD\times\text{bytes\_per\_element}.
\]

例如 B=1、T=8192、D=1024、R=4、BF16，状态张量本身约 80 MiB。该数不包括 stack/FP32 临时量、Core 激活、梯度、KV 和优化器，也不全是相对原 multi-exit 训练的额外开销。

若把 T 扩成 1M，history 当然随之增大。终局设计中 1M 是外部可检索资料，不能因为 Working Memory 小就忽略 backbone 实际处理的 T。

## 18.4 历史路由计算

单轮读 N_r 个来源，大致为 `O(B*T*N_r*D)`；全 R 轮累积为 `O(B*T*D*R²)`。

R=4 时可接受的假设不应无条件推广到 R=32/64。届时应限制历史数量、分块存储或使用 fused routing，并重新测量，不仅看理论 FLOPs。

## 18.5 Reader 计算

对于独立 Reader，query 长度 T、memory 长度 M、attention 总宽度 D_a：QK 与 AV 两次矩阵乘的前向开销约：

\[
4BTMD_a.
\]

另计 Q/O 投影、memory K/V 投影与 Reader Gate。

注意是 `T*M`，不是自动等于 `M²`。decode 时 T=1；prefill 或训练时 T 可能很大。

Memory KV 未变化时，K/V 投影可跨 loop 复用；Q/O、attention 和 Gate 仍要逐轮执行。检索候选重排也可能需要编码很多最终未入 Hot Memory 的块，其成本要单独记。

## 18.6 推荐性能统计

至少记录：

```text
TTFT、decode tokens/s、端到端 latency
实际执行 Core rounds 与样本选中 rounds
global retrieval / warm rescore 次数
unique memory tokens / slots、预算上限
memory K/V cache hit rate 与重投影次数
history read 时间、Reader 时间、Gate 时间
peak allocated / reserved memory
实际 token 数、有效监督 token 数、padding 比例
```

异步 GPU 计时使用正确的同步/CUDA events。训练与推理不同 backend、batch、精度不能混在一张 speedup 表里。

# 19. 推荐配置与版本管理

以下是终局模块配置示例，不是已经被现有 CLI 支持的命令。需先在 config/parser 中接入。

```yaml
architecture_version: minimind_x_residual_v2

# 以实际 checkpoint 的维度和层数为准，不在升级时偷偷改变模型规模。
model:
  inherit_backbone_config: true
  max_loop: 4
  loop_embedding: false
  mhc_enabled: false

loop_mhar:
  enabled: true
  heads: 4
  shared_query_across_loops: true
  key_norm: full_width_rmsnorm
  query_init: zeros
  latest_source_bias_init: 4.0
  history_strength: 1.0
  detach_history: false

recurrent_update:
  enabled: true
  granularity: token_channel
  low_rank: 64
  final_weight_init: zeros
  final_bias_init: 4.0
  gate_compute_dtype: float32
  update_progress: 1.0
  state_noise_std: 0.0
  gate_noise_std: 0.0

memory_reader:
  enabled: true
  placement: after_first_core_self_attention
  reader_core_layer_ids: [0]
  query_heads: 8
  kv_heads: 2
  head_dim: 64
  gate_granularity: token_scalar
  gate_rank: 32
  residual_scale_init: 0.01       # 旧模型严格迁移可设 0
  separate_qkv_projections: true
  apply_token_rope_to_memory_qk: false
  dropout: 0.0
  null_for_empty_queries: true

memory:
  inherit_existing_budget_list: true
  max_memory_tokens_runtime: 8192 # 示例运行上限；不是硬编码结构常数
  deduplicate_by_source_id: true
  track_memory_version: true
  train_projection_with_grad: true
  cache_across_training_steps: false

controller:
  collect_logits_during_full_unroll: true
  detach_gate_statistics_for_halt: true
  retrieval_actions: [reuse, rescore_warm, global_refresh]
  mask_unavailable_actions: true
  query_visibility: prompt_or_current_prefix_only
  force_stop_last_round: true
  retrieve_after_stop: false

cache:
  core_layout: layer_loop_token
  prefill_policy: fixed_cap
  missing_depth_policy: last_valid_per_token
  preserve_past_deep_slots: true
  past_memory_update_policy: append_only

loss:
  answer_weight: 1.0
  halt_weight: 0.1
  retrieval_action_weight: 0.1
  budget_weight: 0.1
  retrieval_evidence_weight: 0.1
  reader_aux_weight: 0.0
  cost_weight_start: 0.0
  gate_mean_regularization: 0.0
  gate_halt_equality_regularization: 0.0
```

配置校验至少包含：D 被 MHAR heads 整除；Reader query_heads 被 kv_heads 整除；T/位置编码范围合法；Reader 插入层存在；budget 受合法 action mask 限制；memory 来源维度与 Bridge 输出一致。

现有 CLI 的显式默认 `hidden_size=768 / num_hidden_layers=8` 会覆盖 preset，容易与 preset 的 zone 数发生冲突。[C10] 升级时应把非显式参数默认改为 `None`，只让用户真正指定的值覆盖已加载 config。

checkpoint 至少保存：模型权重、完整 config、architecture/cache 版本、optimizer/scaler/scheduler、各 rank RNG、数据进度、warm-up 进度及 source/tokenizer/Bridge manifest。不只保存一个 half 精度 state_dict 然后依靠 CLI 重建配置。

# 20. 开发任务清单与定义完成标准

| ID | 开发任务 | 完成标准 |
|---|---|---|
| R-01 | 新增 LoopMHAR | 来源轴 softmax、同 token 路由、H=1 参考等价、history 不被原地修改 |
| R-02 | 新增 UpdateGate | g=0/1 边界正确、bypass 是旧 committed state、支持 active mask |
| R-03 | 新增独立 Reader | 不再拼接 token/memory K/V；空/全 masked memory 严格 no-op |
| R-04 | Core 集成 | 每轮一次 history read、一次完整 Core、一次 update commit |
| R-05 | 输出语义升级 | Router 使用 committed state，不使用 Coda 后池化状态冒充 |
| R-06 | 梯度边界修复 | Reader 投影、retriever 监督 logits 和全部 Router 均有真实梯度路径 |
| R-07 | Query 因果边界 | 换 gold 后缀不改变前面的路由与 next-token logits |
| R-08 | Memory ID/版本 | 去重、有效 mask、缓存按版本失效 |
| R-09 | 可变深度 KV | 保留 R_max 槽、按当前 token 填充、历史深槽不丢失 |
| R-10 | Reference replay | 动态 retrieval/depth 时间线可重放，优化器前后版本明确 |
| R-11 | 训练调度与 loss | 三个模块同一 run 启用；概率收集不导致训练 premature stop |
| R-12 | 计时/成本 | 报告实际执行与逻辑退出的差异，不把关 Gate 算作跳 Core |
| R-13 | checkpoint/CLI | 明确配置继承、missing keys 白名单与架构版本 |
| R-14 | 多卡训练 | DDP/FSDP 正确包装与采样分片；不能仅 torchrun 起多份独立训练 |

任务可以在不同文件并行开发，但**架构只按本文这一套集成**。所有训练启动脚本先通过配置和正确性验收，避免直接投入长周期训练后才发现模块未被优化。

# 21. 必须执行的验收测试

## 21.1 数学不变量

- 单历史来源时，MHAR 输出等于该来源；history_strength=0 时输出等于最新 committed state。
- 每个 token/head 的来源权重之和为 1；改变未来 token 的历史状态不影响更早位置的 MHAR 输出。
- Update force_gate=0/1 分别精确返回 previous/candidate；inactive 位置保持 previous。
- Reader 对 None、零长度、全 masked memory 以及单个 query 无可见 memory 时均严格 no-op，且无 NaN。
- memory padding 内容改变，不影响真实 query 输出。
- BF16 门控、norm、softmax 的数值保持有限。

## 21.2 真实训练接线

在 **真实 `sft_c_step`** 而非专门绕开 controller 的测试函数中，验证 Reader、各 Router、retriever 投影的梯度。

正样本必须需要外部随机事实才能答对；用无 memory/错误 memory 对照排除纯复制或训练样本记忆。

零 scale 初始化允许第一步只有 scale 获得梯度，但后续步 Reader 内部必须开始更新。长期无梯度不能解释为“模型自己选择不使用”。

## 21.3 Cache 与路径一致性

1. 固定 R、固定 memory 下，整段与带 cache 的增量输出在数值容差内一致。
2. 预设深度序列 `4→1→3→2` 时，早停不会删除先前 token 的深轮次 KV。
3. 混合 batch 与逐样本参考执行结果一致，不能由平均 Gate/预算决定所有样本动作。
4. memory 在指定 token/loop 切换版本后，结果与同轨迹参考重放一致。
5. checkpoint 开/关的前向和梯度对齐，不增加检索计数。
6. 同一请求重置、跨请求并发、checkpoint 载入后都不串用 memory/history。

## 21.4 因果与安全

- 同一 prompt 换 gold completion，初始检索不变。
- 同一生成前缀换未来 gold 后缀，当前动作不变。
- sparse baseline 同时验证 attention mask 与 block selector 因果性。
- 未授权来源、错误 source version 不进入 Working Memory。
- 调试日志不输出隐私 payload 或完整 hidden tensors。

## 21.5 任务与效率验收

Gate 直方图不是验收结果。至少比较同一 checkpoint 在固定 R 和自适应策略下的：

```text
实际任务正确率/通过率
错误证据导致的回退幅度
无关 memory 的影响
相同预算下的性能
相同质量下的 latency / executed FLOPs
Gate、Memory、History 的使用统计
```

R 增加后偶尔退化并不自动证明架构错误，但必须报告。若声称缓解 overthinking，须报告“较浅正确、较深转错”的频率及置信区间，而不是只画 Gate 均值递减曲线。

初始工程质量门槛建议以已有可用 checkpoint 为参照：基础能力不出现明显回退、相关证据任务有实际收益、无关/错误证据不大幅放大错误。具体百分比应结合样本量和业务要求设定，不在无数据时承诺固定提升。

# 22. 交付状态、风险与最终结论

## 22.1 本次已经交付的内容

- 一套明确的组合架构和唯一默认插入位置。
- Loop-MHAR、UpdateGate、Reader 的公式、形状、初始化与 API。
- 现有代码需要改动的位置与真实梯度/因果性前置修复。
- 动态检索、可变 Memory、Halt 和 cache 的语义契约。
- 训练与迁移课程、损失接线和效率统计口径。
- 三个独立 PyTorch 模块，以及 CPU 回归测试。

## 22.2 本次没有声称完成的事项

尚未把参考代码合并进原仓库；未运行原仓库声明的 46 个测试；未执行完整 trainer、多卡、动态解码 cache replay 或 1M 上下文训练；未验证该组合相对原版的能力提升。

配套参考测试不依赖缺失的 `model/model_minimind.py`，因此不会用替代 backbone 冒充原模型集成成功。该文件以及原项目完整依赖、配置和训练数据，需要在实际仓库内完成最终接入。

## 22.3 主要风险和处理原则

| 风险 | 不正确的处理 | 推荐处理 |
|---|---|---|
| Gate 长期接近 1 | 强制惩罚 Gate | 先判断是否确实需要大更新，再查训练覆盖和梯度 |
| Gate 接近 0、能力差 | 直接据此提前停止 | 排查饱和、零初始化阻塞、损失冲突、错误固定点 |
| MHAR 只读最新状态 | 认定没有训练 | 与任务表现对照；历史确实可能没有增益 |
| Reader 忽略 memory | 一律强制打开 | 使用必须依赖外部事实的任务，检查投影梯度和对齐 |
| 新模块使速度下降 | 用理论 FLOPs 掩盖 | 分项 profiler，减少复制/重投影，区分 prefill/decode |
| 动态路径训练不稳定 | 把所有策略改成随机标签 | 正确的同前缀反事实监督、课程和完整 replay |
| 无 memory 基线回退 | 宣称 Gate 理论保真 | 做兼容模式回归与迁移对齐，限定保真条件 |

**最终方案可以概括为：先选择从历史读什么，再独立读取外部证据，随后生成候选，最后有控制地写回状态。**

这次改造不是“增加更多 Loop”，而是让既有 Loop 的信息读取、知识注入和状态提交分别受控。mHC 暂不加入，三个模块也不互相冒充正确性判别器。真正的落地标准是：路径因果、梯度接通、缓存可重放、任务有用、成本可测。

---

# 附录 A. 三个核心模块的独立 PyTorch 参考实现

以下代码是本方案的参考实现，文件名 `modules.py`。它不是完整 MiniMind-X 模型：self-attention、检索策略、可变深度 token cache 与训练器集成按正文完成。参考代码使用显式 stack 和 GQA head repeat 以便阅读和测试；性能优化应在等价性验证后替换。

```python
"""MiniMind-X residual-v2 reference primitives.

Independent PyTorch modules, NOT a replacement for the full MiniMind-X model.
Token self-attention, retrieval policy, decoder KV management and trainers remain
integration work. Masks use True = visible/valid. No cross-request state lives here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class StableRMSNorm(nn.Module):
    """RMS statistic and multiplication in FP32; result follows input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class LoopMHAR(nn.Module):
    """Read SAME-TOKEN states across loops, not different token positions.

    history = [anchor, committed_loop_1, ..., committed_loop_(r-1)].
    All states have [B, T, D]. The last entry MUST be the current committed state.
    A shared pseudo-query is partitioned across feature heads. This is a proposed
    loop-boundary adaptation of depth-wise MHAR, not the paper's full architecture.
    """

    def __init__(self, dim: int, heads: int = 4, latest_bias: float = 4.0) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads")
        self.dim, self.heads = dim, heads
        self.norm = StableRMSNorm(dim)
        self.query = nn.Parameter(torch.zeros(heads, dim // heads))
        # Rows: anchor, older committed state, latest committed state.
        self.type_bias = nn.Parameter(torch.zeros(3, heads))
        with torch.no_grad():
            self.type_bias[2].fill_(latest_bias)

    def forward(self, history: Sequence[Tensor], strength: float = 1.0
                ) -> tuple[Tensor, Tensor]:
        if not history or not 0.0 <= strength <= 1.0:
            raise ValueError("nonempty history and strength in [0,1] required")
        current = history[-1]
        if current.ndim != 3 or current.shape[-1] != self.dim:
            raise ValueError("history tensors must have shape [B,T,dim]")
        if any(x.shape != current.shape for x in history):
            raise ValueError("all history tensors must share shape")
        if any(x.device != current.device or x.dtype != current.dtype for x in history):
            raise ValueError("all history tensors must share device and dtype")
        values = torch.stack(list(history), dim=0)  # [N,B,T,D]; reference allocation
        n, b, t, _ = values.shape
        keys = self.norm(values).float().reshape(n, b, t, self.heads, -1)
        # Full-width RMSNorm precedes the head split. No 1/sqrt(d_head) here.
        logits = torch.einsum("he,nbthe->nbth", self.query.float(), keys)
        roles = torch.ones(n, dtype=torch.long, device=current.device)
        roles[0], roles[-1] = 0, 2
        logits = logits + self.type_bias[roles].float()[:, None, None, :]
        weights = logits.softmax(dim=0)  # normalize LOOP/SOURCE axis only
        vh = values.float().reshape(n, b, t, self.heads, -1)
        mixed = torch.einsum("nbth,nbthe->bthe", weights, vh).reshape(b, t, self.dim)
        out = current.float() + strength * (mixed - current.float())
        return out.to(current.dtype), weights


class RecurrentUpdateGate(nn.Module):
    """Commit h_next = h_prev + g * (candidate - h_prev).

    The bypass is the PRE-ROUND committed state, NOT the history-mixed read.
    force_gate exists only for exact compatibility / invariant tests.
    """

    def __init__(self, dim: int, rank: int = 64, bias: float = 4.0) -> None:
        super().__init__()
        if dim <= 0 or rank <= 0:
            raise ValueError("dim and rank must be positive")
        self.dim = dim
        self.h_norm, self.e_norm, self.delta_norm = (
            StableRMSNorm(dim), StableRMSNorm(dim), StableRMSNorm(dim))
        self.fc1 = nn.Linear(3 * dim, rank, bias=True)
        self.fc2 = nn.Linear(rank, dim, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, bias)

    def forward(self, previous: Tensor, anchor: Tensor, candidate: Tensor,
                active: Optional[Tensor] = None, force_gate: Optional[float] = None
                ) -> tuple[Tensor, Tensor]:
        if previous.shape != candidate.shape or previous.shape != anchor.shape:
            raise ValueError("previous, anchor, candidate must share [B,T,D]")
        if previous.ndim != 3 or previous.shape[-1] != self.dim:
            raise ValueError("invalid hidden shape")
        if active is not None and (active.dtype != torch.bool or active.shape != previous.shape[:2]):
            raise ValueError("active must be bool [B,T]")
        if force_gate is None:
            features = torch.cat([self.h_norm(previous), self.e_norm(anchor),
                                  self.delta_norm(candidate - previous)], dim=-1)
            gate = self.fc2(F.silu(self.fc1(features))).float().sigmoid()
            updated = (previous.float() + gate * (candidate.float() - previous.float())).to(previous.dtype)
        else:
            if not 0 <= force_gate <= 1:
                raise ValueError("force_gate must be in [0,1]")
            gate = torch.full_like(previous, force_gate, dtype=torch.float32)
            if force_gate == 0:
                updated = previous
            elif force_gate == 1:
                updated = candidate
            else:
                updated = (previous.float() + gate * (candidate.float() - previous.float())).to(previous.dtype)
        if active is not None:
            updated = torch.where(active[..., None], updated, previous)
        return updated, gate


@dataclass(frozen=True)
class MemoryKV:
    """Independent memory K/V. Cache owner must track version / optimizer step."""
    key: Tensor    # [B,Hkv,M,Dhead]
    value: Tensor  # [B,Hkv,M,Dhead]
    valid: Tensor  # [B,M], bool


class GatedMemoryReader(nn.Module):
    """Separate cross-attention residual; neither memory K nor query uses RoPE.

    Use once per core pass, after the FIRST self-attention residual and before its
    FFN. Do not additionally concatenate memory K/V into token self-attention.
    """

    def __init__(self, dim: int, memory_dim: int, query_heads: int = 8,
                 kv_heads: int = 2, head_dim: int = 64, gate_rank: int = 32,
                 scale_init: float = 0.01, dropout: float = 0.0) -> None:
        super().__init__()
        if min(dim, memory_dim, query_heads, kv_heads, head_dim, gate_rank) <= 0:
            raise ValueError("dimensions and heads must be positive")
        if query_heads % kv_heads or not -1 < scale_init < 1:
            raise ValueError("query_heads must divide by kv_heads; scale_init in (-1,1)")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.dim, self.memory_dim = dim, memory_dim
        self.query_heads, self.kv_heads, self.head_dim = query_heads, kv_heads, head_dim
        self.dropout = dropout
        self.q_norm, self.m_norm = StableRMSNorm(dim), StableRMSNorm(memory_dim)
        self.q_proj = nn.Linear(dim, query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(query_heads * head_dim, dim, bias=False)
        self.read_norm = StableRMSNorm(dim)
        self.gate_fc1 = nn.Linear(2 * dim, gate_rank)
        self.gate_fc2 = nn.Linear(gate_rank, 1)
        nn.init.zeros_(self.gate_fc2.weight)
        nn.init.zeros_(self.gate_fc2.bias)
        # Random/nonzero o_proj + zero scalar = exact adapter no-op, but scalar
        # still gets gradient. Never zero BOTH o_proj and scale.
        self.scale_logit = nn.Parameter(torch.tensor(math.atanh(scale_init)))

    def project_memory(self, hidden: Tensor, valid: Tensor) -> MemoryKV:
        if hidden.ndim != 3 or hidden.shape[-1] != self.memory_dim:
            raise ValueError("memory hidden must have shape [B,M,memory_dim]")
        if valid.dtype != torch.bool or valid.shape != hidden.shape[:2]:
            raise ValueError("valid must be bool [B,M]")
        b, m, _ = hidden.shape
        # Mask before normalization/projection; padded values carry no content.
        clean = torch.where(valid[..., None], hidden, torch.zeros_like(hidden))
        mh = self.m_norm(clean)
        k = self.k_proj(mh).reshape(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mh).reshape(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        return MemoryKV(k, v, valid)

    def forward(self, hidden: Tensor, kv: Optional[MemoryKV],
                query_valid: Optional[Tensor] = None,
                allowed: Optional[Tensor] = None) -> tuple[Tensor, dict[str, Tensor]]:
        b, t, d = hidden.shape
        if d != self.dim:
            raise ValueError("hidden width mismatch")
        if query_valid is None:
            query_valid = torch.ones(b, t, dtype=torch.bool, device=hidden.device)
        if query_valid.dtype != torch.bool or query_valid.shape != (b, t):
            raise ValueError("query_valid must be bool [B,T]")
        if kv is None or kv.key.shape[-2] == 0:
            return hidden, {"gate": hidden.new_zeros(b, t, 1),
                            "delta": torch.zeros_like(hidden)}
        if kv.key.shape[0] != b or kv.value.shape != kv.key.shape:
            raise ValueError("memory K/V batch or shape mismatch")
        m = kv.key.shape[-2]
        visible = kv.valid[:, None, :].expand(b, t, m) & query_valid[..., None]
        if allowed is not None:
            if allowed.dtype != torch.bool or allowed.shape != (b, t, m):
                raise ValueError("allowed must be bool [B,T,M]")
            visible = visible & allowed
        any_real = visible.any(dim=-1, keepdim=True)  # [B,T,1]
        # NULL is visible only when this query has no valid real memory keys.
        visible = torch.cat([visible, ~any_real], dim=-1)[:, None, :, :]
        null_k = kv.key.new_zeros(b, self.kv_heads, 1, self.head_dim)
        k = torch.cat([kv.key, null_k], dim=-2)
        v = torch.cat([kv.value, torch.zeros_like(null_k)], dim=-2)
        rep = self.query_heads // self.kv_heads
        # Reference path; optimized deployment can use supported GQA kernels.
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        q = self.q_proj(self.q_norm(hidden)).reshape(b, t, self.query_heads, self.head_dim).transpose(1, 2)
        read = F.scaled_dot_product_attention(
            q, k, v, attn_mask=visible, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0)
        read = read.transpose(1, 2).reshape(b, t, self.query_heads * self.head_dim)
        message = self.o_proj(read)
        gate_features = torch.cat([self.q_norm(hidden), self.read_norm(message)], dim=-1)
        gate = self.gate_fc2(F.silu(self.gate_fc1(gate_features))).float().sigmoid()
        gate = gate * any_real.float()
        delta = self.scale_logit.float().tanh() * gate * message.float()
        delta = torch.where(any_real, delta, torch.zeros_like(delta))
        out = (hidden.float() + delta).to(hidden.dtype)
        out = torch.where(query_valid[..., None], out, hidden)
        return out, {"gate": gate, "delta": delta.to(hidden.dtype)}
```

# 附录 B. 参考实现测试记录

## B.1 本次实际运行范围

测试环境：PyTorch `2.10.0+cpu`；无 CUDA。

执行：

```bash
cd minimind_x_residual_v2_reference
python -m pytest -q
```

结果：

```text
..........................                                               [100%]
26 passed in 1.25s
```

这里的 26 项是本文新增参考模块的测试，不是附件原仓库的 46 项测试，也不包括完整 KV decoder、动态检索训练、多卡或长上下文能力测试。

## B.2 已执行的测试范围

| 测试组 | 已检查的性质 |
|---|---|
| RMSNorm | 零输入有界、无 NaN |
| Loop-MHAR | 单来源恒等、权重沿历史轴归一、η=0 恒等、单头手工公式对齐 |
| 多头路由 | 不同头可以产生不同历史权重 |
| History 因果性 | 改未来 token 的来源不影响更早 token；不原地修改输入来源 |
| Update Gate | 0/1 极限、0.25 插值、+4 bias 近开门但非严格恒等、inactive 位置不更新 |
| Reader no-op | None、M=0、全遮罩、混合 batch 的空 memory 行 |
| Reader mask | padded 内容不可见、每 query 可见性、无效 query 不更新 |
| Reader 初始化 | scale=0 精确 no-op，但 scale 能获得非零梯度 |
| Reader 梯度 | frozen payload 下 Q/K/V/O 有梯度；冻结下游层不阻断上游 Reader 梯度 |
| KV 复用 | 单一训练图内多次读取同一 Memory KV 可反传 |
| 数值与形状 | BF16 有限、非法输入主动报错 |
| 组合路径 | 线性 toy harness 中三模块都有梯度；该 harness 不是语言模型 |

完整测试源码位于配套包的 `test_modules.py`。

## B.3 尚需在真实仓库执行的验收

第 21 节的真实 trainer、可变深度 cache、memory-version replay、未来 gold 泄漏、多卡和性能测试仍是必需项。只有这些通过，才具备启动正式长周期训练的依据。

# 附录 C. 附件源码证据索引

下表使用上传包 `minimind_x_code_bundle(2).md` 的**包内行号**，不是拆分后的 `.py` 行号。该包只包含新增代码，引用到的原始基础模型及背景文档未包含其中，因此本文不声称审查了那些未提供文件。

| 编号 | 位置与包内行号 | 支持的判断 |
|---|---|---|
| C1 | `thinking_core.py`，L1759–L1768；L1945–L1964 | 既有分轮缓存、全展开、Halt 与外部 memory 的设计语义；每轮提交前后需要重新界定 |
| C2 | `MemoryAwareAttention.forward`，L1785–L1814；`project_memory`，L1859–L1875 | 原实现将 memory 与 token K/V 拼接；memory K 未做 token RoPE |
| C3 | `controller.py`，L350–L371、L426–L449 | 选择/记忆路径位于 no_grad；训练 KV 投影也在 no_grad 中；检索 CE 取缓存 scores |
| C4 | `minimind_x.py`，L1065–L1085；`halt_router.py`，L1703–L1722 | Budget 读 Prelude；Halt 计算由 halt 开关控制；pooling 的粒度 |
| C5 | `train_sft_x.py`，L2880–L2919 | 训练使用 input_ids；新 budget sweep 与 retrieval sweep 调用；损失未传 retrieval logits |
| C6 | `retrieval_qa_dataset.py`，L3402–L3423 | input_ids 由问题、答案分隔、gold answer 构成；因此不能整段用于初始检索 |
| C7 | `qsa_retriever.py`，L1535–L1569；`losses.py`，L677–L690 | 固定 Top-K 与 budget_topk 接口；memory_size 为成本常量时不产生策略梯度 |
| C8 | `sparse_attn.py`，L1620–L1647 | 选择候选块后，allow_blocks 与局部 band 相或；需要统一的因果 mask 与因果 selector |
| C9 | `train_sft_x.py`，L2966–L3002；`train_grpo_x.py`，L2357–L2394 | 初始化 distributed 环境后仍需检查模型包装与数据分片；不能以 torchrun 本身证明同步训练 |
| C10 | `config.py`，L179–L193、L224–L238；`train_pretrain_x.py`，L2707–L2714、L2745–L2756 | CLI 显式默认值与 preset 覆盖链、zone 切分校验 |
| C11 | `tests/test_learnability.py`，L5008–L5019 | memory 可学习性测试刻意保留梯度，与真实 controller 的 no_grad 路径不同 |
| C12 | `minimind_x.py`，L1102–L1105；`thinking_core.py`，L1888–L1892 | 现有 cache 输出按本轮 n_run 重建；历史轮数不足采用 clamp，需升级为持久的逐 token 契约 |

包开头声明了 `46 passed`。这是附件作者的验证声明；本次没有独立复跑原仓库测试。[包内 L1–L7]

# 附录 D. 已核实资料与引用边界

**R1 — Attention Residuals**  
Kimi Team，arXiv:2603.15031。用于支持“对历史深度来源做学习加权读取”的设计依据。  
论文：`https://arxiv.org/abs/2603.15031`  
作者仓库：`https://github.com/MoonshotAI/Attention-Residuals`

**R2 — Multi-Head Attention Residuals**  
Cheng Luo、Zefan Cai、Junjie Hu，arXiv:2607.27230，本文核对 v2 的多头公式。用于支持按 feature 子空间独立深度路由。Loop 边界版本、来源类型偏置及迁移强度是本文的项目化扩展，不是其原样实现。  
论文：`https://arxiv.org/abs/2607.27230`  
全文：`https://arxiv.org/html/2607.27230v2`

**R3 — Gated Recurrent Transformers / RecurrentGPT**  
Amr Hegazy、Amr Alanwar、Mostafa Elhoushi，arXiv:2608.15062。用于支持共享 Core、Prelude 条件与逐元素门控的方向。作者模型卡明确介绍近开门 sigmoid 初始化及其训练设计。本文没有把原工作的噪声与完整训练 recipe 搬入默认配置。  
论文编号入口：`https://arxiv.org/abs/2608.15062`  
作者模型卡：`https://huggingface.co/Amr-Hegazy/grt-medium-isoparam`  
作者模型卡：`https://huggingface.co/Amr-Hegazy/grt-large-isoparam`

**R4 — mHC: Manifold-Constrained Hyper-Connections**  
Zhenda Xie 等，arXiv:2512.24880。作为未纳入本版本的多流残差路线参考；不据此宣称本方案具有同样的稳定性保证。  
论文：`https://arxiv.org/abs/2512.24880`

**R5 — PyTorch Autograd mechanics**  
用于确认冻结参数、no-grad mode 与正常梯度传播之间的区别。  
官方文档：`https://docs.pytorch.org/docs/stable/notes/autograd`

**R6 — PyTorch scaled_dot_product_attention**  
用于确认 bool mask 的 `True=可见`、显式 dropout 控制及相关形状。具体 backend 支持与性能须在目标环境验证。  
官方文档：`https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html`

以上资料核实日期：2026-09-21。本文未将任何论文的 benchmark 增益写成 MiniMind-X 的预期保证，也未以一次 CPU 模块测试替代完整系统验证。

---

**设计冻结结论：保留 MiniMind-X 既有终局主线，用 Loop-MHAR 管历史读取、用独立 Reader Gate 管外部证据注入、用 Recurrent Update Gate 管状态提交，再由 Halt/检索控制器决定后续计算。**
