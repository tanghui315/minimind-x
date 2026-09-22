# MiniMind-X
# Training Data & Training Strategy Design

## Adaptive Compute Foundation Model


版本：v1.0


---

# 1. 训练目标


MiniMind-X 不只是训练语言能力。


目标学习：

```text id="9p7zxm"
Read

↓

Remember

↓

Retrieve

↓

Think

↓

Retrieve Again

↓

Think Again

↓

Answer

```


模型需要学会：

1. 长上下文理解
2. 信息检索
3. 动态 Memory 管理
4. Latent Thinking
5. Adaptive Compute


---

# 2. 最大训练挑战


## Challenge 1

没有天然 Retrieval Label


问题：

数据不会告诉模型：

```json
{
"retrieve":true
}
```


---

## Challenge 2

没有 Memory Size Label


不知道：

```text id="y6l7wq"
当前需要：

2K?

8K?

16K?

```


---

## Challenge 3

没有 Thinking Depth Label


不知道：

```text id="x4qv6f"
需要 loop 几次？

```


---

# 3. 核心训练思想


采用：

# Self-generated Supervision


即：

标签不是人工标注。


而由模型自身产生：


```text id="pxq42m"

Question

      |

      v


不同 Memory Budget

不同 Loop Depth

不同 Retrieval Path


      |

      v


Performance Curve


      |

      v


Generate Training Signal

```


---

# 4. 总体训练流程


```text id="m7yl2d"

Stage 0

Base Pretrain

        |

        v


Stage 1

Long Context Pretrain

        |

        v


Stage 2

Instruction SFT

        |

        v


Stage 3

Adaptive Compute Training

        |

        v


Stage 4

RL Optimization


```


---

# 5. Stage 0

# Base Pretraining


目标：

训练基础语言能力。


保持：

MiniMind 原始：

- tokenizer
- next token prediction


数据：

```text id="8q4a8k"
General Corpus

Books

Code

Web

Wikipedia

Synthetic Text

```


规模：

建议：

64M:

1B tokens


200M:

5B tokens


---

# 6. Stage 1

# Long Context Continued Pretraining


目标：

让模型学会：

```text id="0ps9j4"
长距离依赖
```


Context Curriculum:


## Phase A

8K


## Phase B

32K


## Phase C

128K


## Phase D

512K


## Phase E

1M


不要直接：

```text id="bq4jls"
0 → 1M

```


---

# 7. Long Context 数据构造


## 7.1 Document Packing


构造：

```text id="tv1nqo"
多个文档拼接

长度：

32K~1M

```


---

## 7.2 Synthetic Long Dependency


人为制造：

例如：

前面定义：

```text id="gywqf1"
Variable X = 12345

```


500K tokens 后：

要求：

```text id="1j9l3b"
Use X
```


训练：

Long Retrieval。


---

## 7.3 Needle Data


不是只测试。


用于训练：

```text id="5xkz5h"
Information Localization

```


---

## 7.4 Long Code Repository


非常重要。


构造：

```text id="1j7x4h"
Repository

+

Issue

+

Commit History

+

Tests

```


任务：

```text id="s2rv9d"
Find Bug

Generate Patch

Explain Design

```


---

# 8. Stage 2

# Instruction SFT


目标：

让模型具备：

基础 Agent 能力。


数据：

---

## General Instruction


普通：

```text id="t3av6h"
Question

Answer

```


---

## Long Context QA


格式：

```json
{
"context":"1M document",
"question":"...",
"answer":"..."
}
```


---

## Multi-hop Reasoning


例如：


```text id="j7p3zm"
A related B

B related C

C related D

Question D

```


训练：

跨信息组合。


---

## Coding Agent


重点：

```text id="3z4m5m"
Repository

Issue

Patch

Test

```


---

# 9. Stage 3

# Adaptive Compute Training


这是 MiniMind-X 核心。


目标：

学习：

```text id="6a8h1k"
Memory Budget

+

Retrieval Policy

+

Thinking Depth

```


---

# 10. Online Oracle Generation


训练过程中动态生成。


一个 sample:


执行：


## Memory


不同：

```text id="gq9bqv"
1K

2K

4K

8K

16K

```


---

## Thinking


不同：

```text id="y6f7j9"
Loop 1

Loop 2

Loop 3

Loop 4

```


---

得到：


Performance Matrix:


|Memory|Loop|Loss|
|-|-|-|
|1K|1|2.4|
|2K|2|1.5|
|4K|2|0.8|
|8K|3|0.5|
|16K|4|0.49|


---

# 11. Memory Budget Label


定义：


\[
B^*
=
min(B)
\]


满足：


\[
Loss_B
\le
Loss_{max}
+\epsilon
\]


例如：


8K:

0.50


16K:

0.49


那么：

\[
B^*=8K
\]


---

# 12. Thinking Depth Label


定义：

\[
R^*
=
min(R)
\]


满足：


\[
Loss_R
\le
Loss_{max}
+\epsilon
\]


例如：

```text id="8md6y7"
Loop1

wrong


Loop2

wrong


Loop3

correct


Loop4

correct

```


得到：

\[
R^*=3
\]


---

# 13. Retrieval Label


比较：


Without Retrieval:

\[
L_0
\]


With Retrieval:

\[
L_r
\]


如果：


\[
L_r+\epsilon<L_0
\]


则：


```text id="6j27wt"
Need Retrieval = True

```


---

# 14. Adaptive Controller Training


三个 Router。


---

# Memory Router


输入：

\[
h
\]


输出：

\[
P(B)
\]


类别：

```text id="5t3aj6"
1K

2K

4K

8K

16K

```


Loss:


\[
CE(B,B^*)
\]


---

# Retrieval Router


输出：

\[
P(retrieve)
\]


Loss:


\[
BCE
\]


---

# Thinking Router


输出：

Continue/Stop


Loss:


\[
BCE
\]


---

# 15. Joint Training Objective


最终：


\[
L=
L_{LM}
+
\alpha L_M
+
\beta L_R
+
\gamma L_T
+
\lambda C
\]


其中：


LM:

语言能力


M:

Memory Budget


R:

Retrieval


T:

Thinking


C:

Compute Cost


---

# 16. Stage 4

# RL Optimization


目标：

优化：

```text id="u0kl8y"
Accuracy

vs

Compute

```


Reward:


\[
R=
R_{answer}
-
\lambda_1Memory
-
\lambda_2Retrieval
-
\lambda_3Loop
\]


---

# 17. RL 场景


## Coding


Reward:


```text id="m2c6xl"
Test Pass

+

Patch Quality

-

Compute Cost

```


---

## Math


Reward:


```text id="5hkm9k"
Correct Answer

-

Thinking Steps

```


---

## Agent


Reward:


```text id="q7p44x"
Task Success

-

Tool Calls

-

Memory Usage

```


---

# 18. 数据配比建议


最终 SFT/RL 数据：

|类型|比例|
|-|-:|
|General Instruction|30%|
|Coding Agent|25%|
|Long Context QA|20%|
|Math Reasoning|15%|
|Planning Agent|10%|


---

# 19. 特殊训练数据


## Retrieval Decision Data


生成：


```text id="8c8m7v"
Question

Context

Candidate A

Candidate B


Need Retrieve?

Why?

```


---

## Memory Management Data


训练：

```text id="6j7m82"
Keep

Drop

Replace

```


例如：


原 Memory:

A B C D


新增：

E F


模型选择：

Keep:

A C E


---

# 20. 推荐训练策略


不要一次打开所有 Router。


采用 Curriculum。


---

## Phase 1


关闭：

- Memory Router
- Retrieval Router
- Thinking Router


只训练：

模型能力。


---

## Phase 2


打开：

Thinking Router


学习：

什么时候继续思考。


---

## Phase 3


打开：

Retrieval Router


学习：

什么时候找信息。


---

## Phase 4


打开：

Memory Router


学习：

需要多少信息。


---

# 21. 最终推理流程


```text id="n3r7pp"

Input


 ↓


Initial Retrieval


 ↓


Memory Allocation


 ↓


Thinking Loop


 ↓


Check:


Need Retrieve?

Need More Memory?

Need More Thinking?


 ↓


Repeat


 ↓


Answer

```


---

# 22. 评测体系


## 能力


- MMLU
- GSM8K
- HumanEval
- MBPP


---

## Long Context


- Needle
- Multi Needle
- LongBench
- InfiniteBench


---

## Agent


- SWE-bench
- ToolBench


---

## Efficiency


记录：


```text id="z4t8am"
Average Memory

Average Retrieval

Average Loop

Latency

FLOPs

```


---

# 23. 最终目标


训练出：

不是：

```text id="a5jx7m"
一个固定计算 Transformer
```


而是：


```text id="jq3y7m"
一个会管理自己计算资源的模型


Read

↓

Remember

↓

Think

↓

Search

↓

Think Again

↓

Answer

```


---

# 24. 核心研究问题


MiniMind-X 最终验证：


\[
Can\ a\ model\ learn\ its\ own\ optimal\ computation\ path?
\]


即：

模型是否可以自主决定：


- 读多少？
- 记什么？
- 搜几次？
- 想多久？


---

# End

MiniMind-X Training Strategy

Adaptive Intelligence