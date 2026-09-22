# MiniMind-X
# Adaptive Compute Foundation Model

## 1M Long Context + Dynamic Retrieval + Adaptive Working Memory + Adaptive Thinking


版本：v1.0


---

# 1. 项目定位


MiniMind-X 不是 MiniMind 的简单扩展。

目标：

构建一个面向下一代 Agent / Reasoning / Long Context 场景的小型基础模型研究平台。


核心问题：

传统 Transformer：

```
Input
 ↓
固定 Attention
 ↓
固定 FFN
 ↓
固定 Layer Depth
 ↓
Output
```


存在三个浪费：

## 1. Context 浪费

所有上下文全部进入计算：

```
1M tokens
全部 Attention
```

但是实际任务只需要其中少量信息。


---

## 2. Compute 浪费

所有问题使用固定深度：

```
简单问题:
32 layers

复杂推理:
32 layers
```

计算预算完全一样。


---

## 3. Memory 浪费

模型没有区分：

```
长期知识

当前任务需要的信息

当前正在思考的信息
```


---

# 2. 核心理念


MiniMind-X 提出：

# Adaptive Compute Allocation


模型动态分配：

```
Context Budget

+
Memory Budget

+
Thinking Budget
```


最终：

```
             Problem

                |

                v


        Adaptive Controller


       /          |          \


      /           |           \


 Read More     Think More    Think Where


 Memory        Depth         Region



```


---

# 3. 总体架构


```
                         1M Context

                              |

                              v


                  Long-Term Memory Store

                              |

                              |

                   Dynamic Retrieval

                              |

                              v


                 Adaptive Working Memory

                        1K~16K


                              |

                              v


                    Prelude Encoder


                              |

                              v


              +-----------------------------+

              |    Deliberation Core        |

              |                             |

              |       L4 → L5 → L6          |

              |                             |

              |          Loop               |

              |        1 ~ N                |

              |                             |

              +-----------------------------+


                              |

                              v


                         Coda


                              |

                              v


                         Output

```


---

# 4. Transformer 三段式结构


MiniMind-X 不采用全层 Loop。


采用：

```
Prelude

↓

Deliberation Core

↓

Coda
```


---

## 4.1 Prelude


负责：

- token understanding
- context encoding
- semantic grounding


例如：

```
L1

L2

L3

```


特点：

```
No Loop

No Deep Reasoning
```


---

## 4.2 Deliberation Core


这是模型的 Thinking Zone。


例如：

```
L4

L5

L6
```


执行：

```
[L4 → L5 → L6] × R
```


其中：

```
R = Adaptive Thinking Depth
```


例如：

简单：

```
R=1
```


复杂：

```
R=4
```


---

## 4.3 Coda


负责：

- output formation
- token prediction


例如：

```
L7

L8

```


不参与 Loop。



---

# 5. Long Context Architecture


目标：

支持：

```
1M Context
```


但不采用：

```
1M Full Attention
```


采用：

```
Long Memory

+

Sparse Retrieval

+

Working Memory
```


---

# 6. Long-Term Memory


保存：

```
全部 Context

1M tokens
```


例如：

```
代码仓库

长文档

会议记录

知识库

```


它不是每一步都参与推理。


---

# 7. Sparse Retrieval


负责：

从：

```
1M Context
```

选择：

```
Relevant Information
```


采用：

## Hierarchical Sparse Retrieval


结构：

```
1M Tokens

      |

      v


Coarse Index

      |

      v


Candidate Blocks

16K~32K

      |

      v


Fine Selection

      |

      v


Working Memory

```


---

# 8. Sparse Attention 设计


不采用每层完整 Sparse Attention。


采用：

# Retrieval Attention


职责：

```
寻找信息
```


而不是：

```
每层处理所有信息
```


推荐结构：

```
Compression Index

+

Micro Block Selection

+

Local Context

```


类似：

```
QSA / NSA 思路
```


---

# 9. Adaptive Working Memory


## 9.1 定义


Working Memory:

不是固定：

```
8K
```


而是：

```
B <= Bmax
```


例如：

```
1K

2K

4K

8K

16K
```


---

# 9.2 Memory Hierarchy


```
                Long Context

                    1M


                     |

                     v


              Candidate Memory

                 16~32K


                     |

                     v


             Working Memory

                 1~16K


                     |

                     v


             Thinking Core

```


---

# 10. Working Memory 管理


Memory Manager:

负责：


## ADD

加入新信息


## KEEP

保持重要信息


## EVICT

删除无用信息


## REPLACE

替换低价值内容


---

# 11. Memory Score


每个 Memory Block:

计算：

\[
Score_i
\]


包括：

## Relevance

当前任务相关度


## Attention Usage

历史使用频率


## Novelty

新信息价值


## Redundancy

重复程度


## Persistence

历史重要性



最终：

\[
Score_i
=
f(Rel,Use,Novelty,Redundancy)
\]


---

# 12. Dynamic Retrieval


核心：

不是：

```
Retrieve once
↓

Generate
```


而是：

```
Retrieve

↓

Think

↓

Retrieve Again

↓

Think

↓

Answer
```


---

# 13. Retrieval Controller


每轮 Thinking 后：

得到：

\[
h_r
\]


两个 Router：

---

## Thinking Router


问题：

```
还需要继续思考吗？
```


输出：

```
Continue / Stop
```


---

## Retrieval Router


问题：

```
当前信息够吗？
```


输出：

```
Retrieve / Reuse
```


---

# 14. Dynamic Retrieval Flow


```
                 Hidden State


                       |

          +------------+------------+

          |                         |


          v                         v


   Need Think?              Need Retrieve?


          |                         |


          |                         |


          v                         v


      Next Loop             Sparse Retrieval


                                      |

                                      v


                              Update Memory


```


---

# 15. Adaptive Thinking


目标：

学习：

```
How Long Should I Think?
```


不是固定：

```
4 loops
```


而是：

```
1~N loops
```


---

# 16. Thinking Loop


例如：

```
Working Memory

       |

       v


L4

↓

L5

↓

L6

       |

       ↺

```


每轮：

```
update hidden state

update reasoning state

```


---

# 17. Adaptive Compute Objective


最终优化：

\[
L=
L_{task}
+
\lambda_m C_m
+
\lambda_r C_r
+
\lambda_d C_d
\]


其中：

---

## Task Loss


答案正确。


---

## Memory Cost


\[
C_m=|WorkingMemory|
\]


---

## Retrieval Cost


\[
C_r=N_{retrieval}
\]


---

## Thinking Cost


\[
C_d=N_{loop}
\]


---

目标：

```
最低计算

保持最高能力
```


---

# 18. Unified Adaptive Controller


最终模型拥有：

```
                Hidden State


                     |

                     v


             Adaptive Controller


       +-------------+--------------+


       |             |              |


       v             v              v


 Memory Size   Retrieval   Thinking Depth


```


三个问题：

---

## 我要看多少？

Memory Budget


---

## 我要找什么？

Retrieval Policy


---

## 我要想多久？

Thinking Depth


---

# 19. Attention 设计


最终：

```
Prelude:

Local Attention


↓

Retrieval Layer:

QSA-like Sparse Attention


↓

Thinking Core:

Working Memory Attention


↓

Coda:

Local Attention

```


---

# 20. 不在 Thinking Loop 内重复执行 Global Retrieval


错误：

```
[L4 QSA

 L5 QSA

 L6 QSA]

×4

```


问题：

```
1M retrieval 被重复执行
```


---

正确：

```
1M Context

↓

Retrieve

↓

Working Memory


↓

[L4 L5 L6]

×4 Loop

```


只有：

```
Need More Info

```

才再次 Retrieval。


---

# 21. 最终计算流程


```
Input

 |

 v


Long Context Encoding


 |

 v


Initial Retrieval


 |

 v


Working Memory


 |

 v


Prelude


 |

 v


Loop 1:

Thinking Core


 |

 v


Need Retrieval?


 |

 +--------+

 |        |

No       Yes

 |        |

 |        v

 |    Retrieval

 |

 v


Loop 2


 |

...


 |

 v


Coda


 |

 v


Answer

```


---

# 22. 训练方案


## Pretrain


目标：

学习：

- Long Context Representation
- Memory Retrieval
- Loop Stability


训练：

```
Context Length Curriculum

8K

32K

128K

512K

1M

```


---

# 23. SFT


数据：

增加：

## Long Context QA


## Code Repository Debug


## Multi-hop Reasoning


## Planning


## Agent Task


---

# 24. Reasoning 数据


重点：

```
数学

代码

多跳问题

复杂规划

```


用于训练：

```
Thinking Depth
```


---

# 25. Memory 数据


构造：

```
长文档

长代码

知识库

```


训练：

```
Retrieval

Memory Update

```


---

# 26. 评价指标


## Accuracy


- GSM8K
- BBH
- HumanEval
- LongBench


---

## Context Ability


- Needle Retrieval
- Multi Needle
- Long Code
- Multi-hop QA


---

## Efficiency


记录：

```
Average Memory Size

Average Retrieval Count

Average Loop Count

Latency

FLOPs

```


---

# 27. 核心创新点


## Innovation 1

Adaptive Working Memory


不是固定 Context。


---

## Innovation 2

Dynamic Retrieval


模型主动决定什么时候重新搜索。


---

## Innovation 3

Adaptive Thinking


模型决定思考深度。


---

## Innovation 4

Memory + Reasoning Unified Controller


统一优化：

```
Read

Think

Remember

```


---

# 28. 最终 MiniMind-X 定义


MiniMind-X:

不是：

```
Transformer + Bigger Context
```


而是：

```
Adaptive Cognitive Architecture
```


核心循环：

```
Read

↓

Remember

↓

Think

↓

Need More Information?

↓

Retrieve

↓

Think Again

↓

Answer

```


---

# 29. 最终目标


构建：

一个小规模但具备未来大模型核心能力验证平台：

```
Long Context

+

Sparse Retrieval

+

Dynamic Memory

+

Latent Reasoning

+

Adaptive Compute

```


最终研究问题：

> 一个模型是否可以自主决定：
> “我要读多少信息、记住什么信息、思考多少步？”

如果答案成立，则 MiniMind-X 将成为一种新的 Adaptive Compute Foundation Model 架构范式。