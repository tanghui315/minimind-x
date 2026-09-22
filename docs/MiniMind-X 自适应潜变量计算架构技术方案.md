# MiniMind-X：Adaptive Latent Compute Architecture

## 基于 Recurrence Region + Dynamic Depth 的新一代小模型架构方案

版本：v1.0

---

# 1. 项目背景

## 1.1 研究动机

当前 Transformer 模型的计算方式存在一个明显问题：

> 所有 token、所有输入、所有任务，都使用固定深度和固定计算预算。

例如：

```
Input
 ↓
Layer1
 ↓
Layer2
 ↓
Layer3
 ↓
...
 ↓
Layer32
 ↓
Output
```

无论：

- 简单事实问答
- 复杂数学推理
- 长链代码分析

都必须经过相同数量 Transformer Block。


但是人的思考过程不是这样的：

简单问题：

```
理解
↓
回答
```

复杂问题：

```
理解
↓
分析
↓
推理
↓
验证
↓
修正
↓
回答
```

因此未来模型需要具备：

1. 动态计算深度（Dynamic Depth）
2. 动态计算位置（Dynamic Compute Placement）
3. 动态计算预算分配（Adaptive Compute Allocation）

---

# 2. 核心研究假设

提出：

# Adaptive Latent Compute

核心思想：

> 模型不仅应该学习“需要思考多久”，还应该学习“应该在哪里进行思考”。

传统 MoR：

```
How long to think?
```

MiniMind-X：

```
Where to think?
+
How long to think?
```

---

# 3. 总体架构设计


## 3.1 三阶段 Transformer


整体：

```
                 Input Tokens

                      |
                      v

        +---------------------------+
        |        Prelude             |
        |                            |
        | 输入理解区域                |
        | 语义融合                    |
        | Context Encoding           |
        |                            |
        | No Loop                    |
        +---------------------------+

                      |
                      v


        +---------------------------+
        |    Deliberation Core       |
        |                            |
        |     Adaptive Loop Zone     |
        |                            |
        |  Semantic Refinement       |
        |  Reasoning                 |
        |  Planning                  |
        |                            |
        |  Loop 1~N                  |
        +---------------------------+

                      |
                      v


        +---------------------------+
        |          Coda              |
        |                            |
        | 输出组织                    |
        | LM Prediction              |
        |                            |
        | No Loop                    |
        +---------------------------+

                      |
                      v

                  LM Head

```


---

# 4. 为什么不是全层 Loop


## 4.1 全层 Loop 的问题


传统 Loop Transformer：

```
L1
L2
L3
L4
  ↑
  |
 repeat
```


问题：

### 1. 浅层表示不成熟

低层主要负责：

- token embedding
- 位置关系
- 局部语义

重复计算收益有限。


### 2. 高层接近输出空间

靠近 LM Head：

- token prediction
- logit refinement

过度循环可能导致：

- overthinking
- 表示漂移


因此：

真正适合循环的位置应该是：

```
语义形成之后

↓

推理之前

↓

中间 latent reasoning space

```

---

# 5. MiniMind-X 第一版架构


基于 MiniMind 8 Layer。


原模型：

```
L1
L2
L3
L4
L5
L6
L7
L8
```

改造：

```
              Prelude

L1
L2
L3


              ↓


       Deliberation Core

       L4
       L5
       L6

       ↺ Loop 1~4


              ↓


              Coda


L7
L8

```

---

# 6. Recurrence Zone 设计


## 6.1 固定区域版本


第一阶段：

不引入动态区域。

只验证：

Loop 放在哪里最好。


实验：

|方案|结构|
|-|-|
|Baseline|L1-L8|
|Full Loop|L1-L8 × R|
|Middle Loop|L1-L3 + (L4-L6 × R)+L7-L8|
|Late Middle|L1-L4 +(L5-L6 × R)+L7-L8|


重点比较：

```
Full recurrence

vs

Middle recurrence

vs

Late-middle recurrence

```


---

# 7. Dynamic Depth 设计


## 7.1 Loop 深度


最大：

```
Rmax = 4
```


模型可以：

```
Loop 1

Loop 2

Loop 3

Loop 4

Exit
```


不同输入：

```
简单问题:

R=1


普通问题:

R=2


复杂推理:

R=4

```


---

# 8. Halting Router


不预测：

```
depth=1/2/3/4
```


而预测：

```
continue or stop
```


例如：

目标：

```
Depth=3
```


训练：

```
Loop1:
continue=1


Loop2:
continue=1


Loop3:
continue=0

```


公式：

\[
P(stop|h_r)
\]


---

# 9. 训练方案


## Phase 0：基础 Pretrain


目标：

让模型支持不同计算深度。


训练：

随机 Loop。


例如：

```
sample:

R=1
R=2
R=3
R=4

```


Loss：

\[
L=L_{LM}
\]


无：

- depth label
- router loss


---

# Phase 1：第一轮 SFT


数据：

保持普通 SFT。


例如：

```
Question

Answer

```


不需要：

```
depth:
3

```


训练方式：

随机：

```
R=1~4

```


目标：

建立：

Depth Elasticity


即：

不同 loop 都能工作。


---

# Phase 2：Online Depth Learning


训练过程中实时计算：

```
R1

R2

R3

R4

```


得到：

```
loss curve:

R1 loss = 3.2

R2 loss = 1.8

R3 loss = 0.7

R4 loss = 0.68

```


定义：

最小有效深度：

\[
D^*
=
min(R)
\]

满足：

\[
Loss_R <= Loss_{max}+epsilon
\]


---

# Phase 3：联合训练


最终 Loss：

\[
L=
L_{task}
+
\alpha L_{halt}
+
\beta L_{consistency}
+
\lambda L_{compute}
\]


其中：

## Task Loss

正常：

\[
CE(answer)
\]


## Halting Loss


训练：

continue / stop


## Consistency Loss


深层结果指导浅层：

\[
KL(P_{Rsmall}||P_{Rmax})
\]


## Compute Loss


鼓励少计算：

\[
L_{compute}=E[R]
\]


---

# 10. Reasoning-Aware Training


进一步加入：

## Latent Reasoning Supervision


如果数据存在 CoT：

例如：

```
Step1

Step2

Step3

Answer

```


不训练：

```
depth label

```


而训练：

```
Loop hidden state
对应 reasoning progress

```


目标：

```
Loop1 hidden

≈

Reasoning Step1


Loop2 hidden

≈

Reasoning Step2

```


---

# 11. 数据方案


## 11.1 普通预训练


继续使用：

MiniMind 原始 corpus。


不需要 depth。


---

## 11.2 Reasoning 数据


重点：

### 数学

```
Arithmetic

Algebra

Geometry

Proof

```


### Code

```
Debug

Algorithm

Execution Trace

```


### Multi-hop

```
1 hop

2 hop

4 hop

8 hop

```


### Planning

```
Multi-step task

```


---

# 12. Oracle Depth 生成


不人工标注。


使用模型自身。


流程：


```
Sample

 |
 +---- R=1

 |
 +---- R=2

 |
 +---- R=3

 |
 +---- R=4


Compare performance


Generate Depth

```


得到：

```
Minimum sufficient computation

```


---

# 13. 实验路线


## Experiment 1

Baseline

```
MiniMind Dense

```


---

## Experiment 2

Full Loop

```
All layers recurrence

```


---

## Experiment 3

Middle Loop

```
L4-L6 recurrence

```


---

## Experiment 4

Dynamic Depth

```
Middle Loop

+

Halting Router

```


---

## Experiment 5

Reasoning Aware Loop

加入：

```
CoT step supervision

```


---

## Experiment 6

Adaptive Zone

学习：

```
Where to Loop

+

How Long to Loop

```


---

# 14. 评价指标


## 能力

- Validation Loss
- PPL
- GSM8K
- ARC
- BBH
- HumanEval


---

## 计算效率


记录：

```
Average Loop Number

Average FLOPs

Latency

Tokens/sec

```


重点：

Accuracy / Compute Pareto


---

## Router 指标


观察：

```
Depth distribution

Task difficulty vs depth

Stop accuracy

```


---

# 15. 最终目标模型


MiniMind-X:


```
                 Token


                   |
                   v


             Semantic Encoder


                   |
                   v


          Adaptive Thinking Core


              /          \

        Which Zone?    How Deep?


              \          /


              Loop


              1~N


                   |

                   v


              Answer

```


---

# 16. 核心创新点


## Innovation 1

Recurrence Region

> Loop 不再作用于整个 Transformer，而作用于最适合 latent reasoning 的区域。


---

## Innovation 2

Adaptive Depth

> 模型学习自己的 thinking budget。


---

## Innovation 3

Latent Reasoning

> 用 hidden state recurrence 替代部分显式 CoT。


---

## Innovation 4

Compute Allocation

> 在固定 FLOPs 下动态分配计算。


---

# 17. 第一阶段落地目标


不要追求一次做完。


第一目标：

验证：

```
Middle Loop

>

Full Loop

```


第二目标：

验证：

```
Dynamic Depth

可以减少平均计算

同时保持能力

```


第三目标：

验证：

```
Latent Depth

≈

Reasoning Depth

```


---

# 18. 项目名称建议


候选：

- MiniMind-X
- Adaptive Reasoning Transformer (ART)
- Latent Thinking Transformer (LTT)
- Dynamic Compute Transformer (DCT)
- Recursive Reasoning Network (RRN)

推荐：

# MiniMind-X ART

Adaptive Reasoning Transformer


---

# 总结

MiniMind-X 的核心不是增加更多参数。

而是重新定义：

Transformer 的计算方式。


从：

```
固定层数
固定计算
固定路径
```

变成：

```
动态区域

动态深度

动态计算预算

```


最终目标：

> 让模型像人一样，在简单问题上快速回答，在复杂问题上自动投入更多内部计算。