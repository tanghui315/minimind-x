好的。前面的三份文档解决的是：

1. **架构是什么**
2. **模块怎么实现**
3. **训练流程怎么走**

这一份解决的是最关键的问题：

> **如何证明 MiniMind-X 不是“堆模块”，而是真正产生 Scaling Advantage。**

也就是类似 DeepSeek / Qwen technical report 里面最重要的部分：

* Scaling Curve
* Compute Efficiency
* Ablation Matrix
* Pareto Frontier
* Architecture Validation

核心研究问题：

$$
\boxed{
Adaptive\ Compute
是否能在相同FLOPs下超过Fixed\ Compute?
}
$$

---

# MiniMind-X Scaling Law 与实验设计方案

## Adaptive Compute Foundation Model

---

# 1. Scaling 研究目标

传统 Scaling Law：

$$
Loss=f(
Parameters,
Tokens,
Compute
)
$$

经典语言模型 Scaling Law 主要关注：

* 模型参数
* 数据量
* 训练计算

损失通常呈幂律下降。([OpenAI][1])

但是 MiniMind-X 增加三个变量：

$$
\boxed{
Compute\ Policy
}
$$

新的 Scaling：

$$
Loss=
f(
N,
D,
C,
M,
R,
Z
)
$$

其中：

| 变量 | 含义               |
| -- | ---------------- |
| N  | 参数量              |
| D  | 训练Token          |
| C  | 训练Compute        |
| M  | Memory Budget    |
| R  | Reasoning Depth  |
| Z  | Reasoning Region |

---

# 2. MiniMind-X 的核心假设

## Hypothesis 1

### Dynamic Memory 优于 Fixed Context

固定：

```text
所有问题:

8K Working Memory
```

动态：

```text
简单:
2K

复杂:
8K
```

假设：

$$
Accuracy_{dynamic}
>
Accuracy_{fixed}
$$

同时：

$$
FLOPs_{dynamic}
<
FLOPs_{fixed}
$$

---

# Hypothesis 2

### Dynamic Thinking 优于 Fixed Depth

传统：

```text
所有问题:

4 Loop
```

MiniMind-X：

```text
Easy:

1 Loop


Hard:

4 Loop
```

目标：

同等能力：

$$
Compute \downarrow
$$

---

# Hypothesis 3

### Dynamic Retrieval 优于 One-shot Retrieval

Baseline:

```text
Retrieve once

↓

Answer
```

MiniMind-X:

```text
Retrieve

↓

Think

↓

Retrieve

↓

Think

```

目标：

Long Context Accuracy ↑

---

# Hypothesis 4

### 联合 Adaptive Compute 存在 Scaling Advantage

最终：

$$
Memory
+
Retrieval
+
Thinking
$$

优于：

单独优化。

---

# 3. 模型 Scaling Plan

不要只训练一个模型。

设计：

## Small

```text
100M
```

用途：

快速验证。

---

## Medium

```text
500M
```

主研究。

---

## Large

```text
1.5B
```

验证 Scaling。

---

# 4. 参数配置建议

## MiniMind-X-100M

```yaml
hidden:
512

layers:
12

thinking_layers:
4

heads:
8

```

---

## MiniMind-X-500M

```yaml
hidden:
1024

layers:
24

thinking_layers:
8

```

---

## MiniMind-X-1.5B

```yaml
hidden:
1536

layers:
32

thinking_layers:
12

```

---

# 5. 为什么增加 Thinking Layer？

因为 MiniMind-X 的计算不是单纯 depth。

而是：

$$
EffectiveDepth
$$

例如：

500M：

物理：

```text
24 layers
```

Thinking Zone：

```text
8 layers
```

Loop:

```text
R=1~4
```

有效深度：

$$
24+8(R-1)
$$

最大：

$$
48 layers
$$

---

# 6. Scaling 实验矩阵

## Baseline

必须有：

---

## Model A

Dense Transformer

```text
Fixed Context

Fixed Depth
```

---

## Model B

MoE

```text
Fixed Context

Fixed Depth
```

---

## Model C

Sparse Attention

```text
Dynamic Context

Fixed Depth
```

---

## Model D

MiniMind-X

```text
Dynamic Context

Dynamic Retrieval

Dynamic Depth

```

---

# 7. 核心比较指标

不能只看 Accuracy。

必须看：

$$
Performance / Compute
$$

---

# 8. Memory Scaling

测试：

```text
Memory Budget

1K

2K

4K

8K

16K

```

记录：

| Memory | Accuracy | FLOPs |
| ------ | -------- | ----- |
| 1K     |          |       |
| 2K     |          |       |
| 4K     |          |       |
| 8K     |          |       |
| 16K    |          |       |

寻找：

Pareto Point。

---

# 9. Thinking Scaling

测试：

Loop:

```text
1

2

3

4

8
```

记录：

| Loop | Accuracy | Latency |
| ---- | -------- | ------- |
| 1    |          |         |
| 2    |          |         |
| 4    |          |         |
| 8    |          |         |

---

# 10. Dynamic Depth 曲线

重点观察：

简单任务：

```text
Loop Distribution

1:80%

2:20%

```

复杂任务：

```text
Loop Distribution

1:10%

2:30%

3:40%

4:20%

```

证明：

模型真的理解任务难度。

---

# 11. Retrieval Scaling

比较：

## Static RAG

```text
Retrieve once
```

---

## Dynamic Retrieval

```text
Retrieve
Think
Retrieve
```

指标：

| 方法      | Accuracy | Retrieval次数 |
| ------- | -------- | ----------- |
| Static  |          |             |
| Dynamic |          |             |

目标：

$$
Accuracy↑
$$

同时：

$$
Retrieval↓
$$

---

# 12. Long Context Scaling

Context:

```text
8K

32K

128K

512K

1M

```

测试：

## Needle

信息定位。

---

## Multi Needle

多个信息组合。

---

## Long QA

理解。

---

## Long Code

Repository。

---

# 13. 最重要实验：Compute Pareto

横轴：

$$
Compute
$$

纵轴：

$$
Accuracy
$$

比较：

```text id="j8wxaf"
Dense

MoE

Sparse

MiniMind-X

```

目标：

MiniMind-X 曲线：

向左上移动。

即：

同样准确率：

计算更低。

或者：

同样计算：

准确率更高。

---

# 14. Scaling Law 拟合

传统：

$$
L=A N^{-\alpha}
$$

MiniMind-X:

增加：

$$
R,M
$$

新的公式：

$$
L
=
A
N^{-\alpha}
D^{-\beta}
R^{-\gamma}
M^{-\delta}
$$

研究：

是否存在：

$$
\gamma,\delta>0
$$

证明：

Thinking Budget 和 Memory Budget 是新的 scaling axis。

---

# 15. Adaptive Compute Scaling

定义：

Effective Compute:

$$
C_{eff}
=
C_{attention}
+
C_{memory}
+
C_{loop}
$$

然后：

比较：

固定：

$$
C=C_{max}
$$

动态：

$$
E(C)<C_{max}
$$

如果：

$$
Accuracy_{dynamic}
\approx
Accuracy_{fixed}
$$

但：

$$
E(C)
<
C_{max}
$$

说明：

模型学会计算分配。

---

# 16. Ablation 设计

这是论文核心。

---

# Ablation A

去掉 Dynamic Memory

变：

固定8K。

观察：

影响。

---

# Ablation B

去掉 Dynamic Retrieval

变：

一次 Retrieval。

---

# Ablation C

去掉 Adaptive Thinking

固定 Loop。

---

# Ablation D

去掉 Thinking Zone

全层 Loop。

---

# Ablation E

去掉 Sparse Attention

Full Attention。

---

# Ablation F

只保留一个 Router

比较：

单控制器 vs 多控制器。

---

# 17. 关键实验

## Experiment 1

Memory-only

问题：

动态 Memory 是否有效？

结构：

```text
Fixed Loop

Dynamic Memory

```

---

## Experiment 2

Thinking-only

结构：

```text
Fixed Memory

Dynamic Loop

```

---

## Experiment 3

Retrieval-only

结构：

```text
Dynamic Retrieval

Fixed Loop

```

---

## Experiment 4

Full MiniMind-X

```text
Dynamic Memory

+

Dynamic Retrieval

+

Dynamic Thinking

```

这是最终模型。

---

# 18. Agent Benchmark

因为 MiniMind-X 本质面向 Agent。

测试：

## SWE

代码修复。

---

## Tool Use

工具调用。

---

## Long Horizon Task

长任务。

指标：

不仅：

Success Rate

还：

```text
Memory Used

Loop Count

Tool Calls

```

---

# 19. 训练成本 Scaling

记录：

每个模型：

| Model | Tokens | GPU | Hours |
| ----- | ------ | --- | ----- |
| 100M  |        |     |       |
| 500M  |        |     |       |
| 1.5B  |        |     |       |

拟合：

$$
Performance=f(Compute)
$$

---

# 20. H20 8卡规划

## Phase 1

100M

目标：

架构验证。

资源：

```text
1~2 H20
```

---

## Phase 2

500M

主实验：

```text
8 H20
```

训练：

10B tokens。

---

## Phase 3

1.5B

最终：

```text
8 H20

30B tokens
```

---

# 21. 最终论文级核心结果

希望得到：

## Figure 1

Architecture

Adaptive Compute。

---

## Figure 2

Scaling Curve

参数 vs Loss。

---

## Figure 3

Compute Pareto

Dynamic vs Fixed。

---

## Figure 4

Memory Scaling

Memory Budget 曲线。

---

## Figure 5

Thinking Scaling

Loop 曲线。

---

## Figure 6

Long Context

1M benchmark。

---

# 22. 最终证明目标

不是证明：

> MiniMind-X 参数更多。

而是证明：

$$
\boxed{
Intelligence
=
Parameter
+
Data
+
Compute\ Allocation
}
$$

模型能力来自：

```text
更多参数

+

更聪明地使用计算资源

```

---

# 23. 最终研究命题

MiniMind-X 希望回答：

> Scaling 的下一维是否不是参数量，而是模型自主分配计算资源的能力？

即：

未来模型 Scaling：

传统：

$$
Parameters↑
$$

未来：

$$
\boxed{
Parameters
+
Memory
+
Reasoning
+
Retrieval
+
Adaptive Compute
}
$$

---

# End

MiniMind-X Scaling Law & Experimental Design
:::

---

我认为到这里，MiniMind-X 已经形成一个完整研究闭环：

```
架构:
Adaptive Compute Transformer

↓

数学:
Memory / Retrieval / Thinking Controller

↓

训练:
Pretrain → SFT → RL

↓

Scaling:
证明 Dynamic Compute 是否形成新 Scaling Axis
```

下一步最关键的是补最后一块：

**《MiniMind-X 与现有 SOTA 架构对比分析（DeepSeek MoE / Qwen3.8-Next / Kimi Linear / Titans / MoR / Huginn）以及创新点定位》**

这个会决定：

* 哪些是已有思想；
* 哪些是真正创新；
* MiniMind-X 应该如何包装成论文/技术报告。

[1]: https://openai.com/index/scaling-laws-for-neural-language-models/?utm_source=chatgpt.com "Scaling laws for neural language models | OpenAI"
