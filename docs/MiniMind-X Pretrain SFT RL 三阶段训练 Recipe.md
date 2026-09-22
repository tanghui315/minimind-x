# MiniMind-X Training Recipe

## Adaptive Compute Foundation Model

### Pretrain → SFT → RL


版本：v1.0


---

# 1. 总体训练路线


MiniMind-X 不采用传统：

```text
Pretrain
 ↓
SFT
 ↓
RLHF
```


而采用：

```text
Stage 1

Foundation Learning

(Base Model)

        ↓


Stage 2

Instruction + Memory + Reasoning Learning

(SFT)


        ↓


Stage 3

Adaptive Compute Optimization

(RL)


```


---

# 2. 模型规模规划


推荐三个规模：

|模型|参数|用途|
|-|-:|-|
|MiniMind-X-S|100M|快速架构验证|
|MiniMind-X-M|500M|主研究模型|
|MiniMind-X-L|1.5B|最终验证|


---

# 3. Stage 1

# Foundation Pretraining


目标：

训练：

- Language Ability
- Context Representation
- Stable Hidden Space


暂时：

关闭：

- Dynamic Retrieval Router
- Memory Controller
- Thinking Router


但是：

保留：

- Long Context Backbone
- Working Memory Interface


---

# 4. Stage1 Architecture


训练模式：

```text id="1v86xc"

Input


 ↓


Prelude


 ↓


Thinking Core

(loop disabled)


 ↓


Coda


 ↓


LM Head


```


原因：

先让 Transformer 学会基础表示。


---

# 5. Pretraining Data


## 5.1 General Corpus


比例：

60%


包含：

- Web
- Books
- Wikipedia
- Papers
- Code


---

## 5.2 Code


比例：

20%


包含：

- GitHub
- Repository
- Documentation
- Issue


---

## 5.3 Synthetic Reasoning


比例：

20%


包含：

- Math
- Logic
- Multi-hop
- Planning


---

# 6. Token Budget


推荐：


## 100M 模型


\[
2B tokens
\]


---

## 500M 模型


\[
10B tokens
\]


---

## 1.5B 模型


\[
30B tokens
\]


---

# 7. Long Context Curriculum


不能直接 1M。


采用：


```text id="7akv8g"

8K

 ↓

32K

 ↓

128K

 ↓

512K

 ↓

1M


```


---

# 8. Long Context Pretrain 数据


重点构造：

## Document Continuation


例如：

一个文档：

```text id="5kj0aa"
100K tokens

```


训练：

预测后续。


---

## Needle Retrieval


随机插入：


```text id="s83k42"
key fact

```


距离：

```text id="1p5avm"
10K

100K

500K

1M

```


任务：

恢复信息。


---

## Repository Context


例如：

整个项目：


```text id="6t9n8f"
src/

test/

docs/

history/

```


任务：

预测：

- bug
- patch
- explanation


---

# 9. Stage1 Loss


标准：

\[
L=L_{next-token}
\]


加入：

Long Context Auxiliary:


\[
L=
L_{LM}
+
\lambda L_{retrieval-aware}
\]


---

# 10. Stage 2

# SFT Training


目标：

让模型学会：

- Instruction Following
- Reasoning
- Agent Behavior


同时开始训练：

- Memory Router
- Thinking Router


---

# 11. Stage2 分三个阶段


## SFT-A

基础 Instruction


打开：

```text id="8cnp7k"
Thinking Core
```


但是：

固定：

\[
R=1
\]


目的：

避免模型依赖 Loop。


---

## SFT-B

Multi-depth Training


随机：

```text id="k2q5pr"
Loop:

1

2

3

4

```


目标：

建立：

Depth Elasticity


即：

不同深度都能工作。


---

## SFT-C

Memory/Retrieval Training


打开：


- Retrieval Router
- Working Memory


---

# 12. SFT 数据


## Instruction


30%


---

## Coding Agent


25%


格式：


```text id="0rfhpi"

Repository

Issue

Context

Patch

Test


```


---

## Long Context


20%


任务：

- QA
- Summarization
- Retrieval


---

## Reasoning


15%


包括：

- Math
- Logic
- Planning


---

## Agent


10%


包括：

- Tool Use
- Multi-step task


---

# 13. SFT Dynamic Compute Label


不人工标。


在线生成。


---

对于一个 sample:


运行：


Memory:

```text id="0uh4x8"
2K

4K

8K

16K

```


Loop:

```text id="2kryc1"
1

2

3

4

```


得到：

Performance Matrix。


---

# 14. Memory Router Training


目标：

预测：

\[
B^*
\]


Loss:


\[
L_M=CE(B,B^*)
\]


---

# 15. Thinking Router Training


目标：

预测：

continue/stop。


例如：

```text id="dhf0e4"
Loop1:

continue


Loop2:

continue


Loop3:

stop

```


Loss:

\[
L_T=BCE
\]


---

# 16. Retrieval Router Training


学习：

是否需要重新 Retrieval。


标签：

比较：

Without Retrieval

vs

With Retrieval


如果：

\[
Gain>\epsilon
\]


label:

```text id="c1c5tw"
Retrieve=True

```


Loss:


\[
L_R=BCE
\]


---

# 17. SFT Final Loss


\[
L=
L_{SFT}
+
\alpha L_M
+
\beta L_T
+
\gamma L_R
\]


---

# 18. Stage3

# Adaptive Compute RL


这是 MiniMind-X 的核心。


目标：

优化：

Accuracy

vs

Compute


---

# 19. RL Algorithm


推荐：

GRPO


原因：

- 不需要 Value Model
- 适合 reasoning
- 成本低


---

# 20. RL Sample


一个问题：


模型生成多个轨迹：


Trajectory 1:


```text id="7sn3ks"
Memory 8K

Loop 4

Answer Correct

```


Trajectory 2:


```text id="0r1f6g"
Memory 4K

Loop 2

Answer Correct

```


Trajectory 3:


```text id="q8n2wb"
Memory 2K

Loop 1

Answer Wrong

```


---

# 21. Reward Design


总 Reward:


\[
R=
R_{task}
-
\lambda_1R_{memory}
-
\lambda_2R_{retrieval}
-
\lambda_3R_{thinking}
\]


---

# 22. Task Reward


## Math


正确：

+1


错误：

0


---

## Coding


Test Pass:


+1


Partial:


0~1


---

## Agent


Task Success:


+1


---

# 23. Compute Penalty


Memory:

\[
P_m=\frac{|W|}{W_{max}}
\]


---

Retrieval:

\[
P_r=N_{retrieve}
\]


---

Thinking:

\[
P_t=\frac{R}{R_{max}}
\]


---

最终：

\[
R=
Accuracy
-
\lambda_mP_m
-
\lambda_rP_r
-
\lambda_tP_t
\]


---

# 24. RL Curriculum


不能一开始压缩计算。


---

## RL Phase1


关闭：

Compute Penalty


目标：

找到正确路径。


---

## RL Phase2


加入：

小 penalty


学习：

减少浪费。


---

## RL Phase3


强化：

Pareto Optimization


---

# 25. 防止 Collapse


## Problem 1


全部：

Loop=1


解决：

Minimum reward constraint。


---

## Problem 2


全部：

Memory=16K


解决：

Memory cost。


---

## Problem 3


全部 Retrieval


解决：

Retrieval penalty。


---

# 26. Final Objective


最终：


\[
\boxed{
Maximize\ Intelligence
\over
Compute
}
\]


---

# 27. H20 8卡训练规划


## Stage1


模型：

500M


GPU:

8×H20


目标：

10B tokens


时间：

约数天~两周


---

## Stage2


SFT:


1B tokens


GPU:

4~8 H20


---

## Stage3


RL:


重点：

Coding + Math


GPU:

8 H20


---

# 28. 最终 Inference


Fast Mode:


```text id="58t8qq"
Memory 2K

Loop 1

```


---

Normal Mode:


```text id="8k3s4h"
Memory 8K

Loop 2

```


---

Deep Mode:


```text id="n7y2sv"
Memory 16K

Loop 4

```


---

# 29. 最终模型能力


MiniMind-X:

不是：

固定 Transformer。


而是：


```text id="u5g4c1"
Read

↓

Retrieve

↓

Remember

↓

Think

↓

Search Again

↓

Think Again

↓

Answer

```


---

# 30. 核心研究假设


验证：

\[
Can\ a\ model\ learn\ its\ own\ computation\ policy?
\]


即：

模型自主决定：

- 看多少？
- 找几次？
- 记多少？
- 想多久？


---

# End

MiniMind-X Training Recipe