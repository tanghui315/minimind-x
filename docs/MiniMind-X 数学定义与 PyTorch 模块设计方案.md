# MiniMind-X
# Mathematical Definition & PyTorch Architecture Design

## Adaptive Compute Foundation Model

版本：v1.0


---

# 1. 总体数学定义


MiniMind-X 定义为：

\[
F(x)=
C(
T(
M(
R(
P(x)
)
)
)
)
\]


其中：

|模块|含义|
|-|-|
|P|Prelude Encoder|
|R|Dynamic Retrieval|
|M|Adaptive Working Memory|
|T|Adaptive Thinking Core|
|C|Coda Decoder|


整体：

```text
Input Tokens

      |
      v

Prelude

      |

Dynamic Retrieval

      |

Working Memory

      |

Thinking Loop

      |

Coda

      |

Output Tokens

```

---

# 2. Transformer Block 划分


假设 MiniMind 原始：

```text
Layer 0~7
```


改造：

```text
Layer 0-2

Prelude


Layer 3-5

Thinking Core


Layer 6-7

Coda

```


定义：

\[
L=P+T+C
\]


其中：

\[
P=3
\]

\[
T=3
\]

\[
C=2
\]


---

# 3. Hidden State 定义


输入：

\[
X\in R^{B\times S\times d}
\]


其中：

B:

batch


S:

sequence length


d:

hidden dimension


例如：

MiniMind:

\[
d=768
\]


---

# 4. Prelude


计算：

\[
H_0=P_\theta(X)
\]


输出：

\[
H_p
\]


即：

```text
Input

[B,S,d]


       |

       v


Prelude


[B,S,d]

```


Prelude 负责：

- token understanding
- semantic encoding
- context grounding


不参与：

- loop
- retrieval decision


---

# 5. Long Context Memory


定义：

长期上下文：

\[
C_L
\]


例如：

\[
|C_L|=1M
\]


形式：

\[
C_L=
\{c_1,c_2,...,c_n\}
\]


每个 block：

\[
c_i\in R^{m\times d}
\]


例如：

block size:

256 tokens


---

# 6. Sparse Retrieval Module


## 6.1 Retrieval Query


根据当前 hidden state：

\[
q=W_q H
\]


其中：

\[
q\in R^d
\]


---

## 6.2 Hierarchical Retrieval


两阶段。


## Stage 1: Coarse


压缩：

\[
C'_L=f_c(C_L)
\]


得到：

\[
N_c
\]


例如：

```text
1M tokens

↓

4000 blocks

```


计算：

\[
score_i=qK_i^T
\]


选择：

\[
TopK_c
\]


---

## Stage 2: Fine


对候选 block：

\[
C_f
\]


进一步：

\[
score_j=qK_j^T
\]


得到：

\[
TopK_f
\]


最终：

\[
W=
Retrieve(C_L,q)
\]


---

# 7. Retrieval Router


决定：

是否需要重新 Retrieval。


输入：

\[
H_t
\]


输出：

\[
p_r
=
\sigma(W_rH_t)
\]


其中：

\[
p_r\in[0,1]
\]


含义：

\[
p_r=P(Retrieve)
\]


判断：

```python
if p_r > threshold:

    retrieve()

else:

    reuse_memory()

```


---

# 8. Adaptive Working Memory


定义：

\[
W_t
\]


最大：

\[
|W_t|\le B_{max}
\]


例如：

\[
B_{max}=16K
\]


---

## Memory Slot


采用 block slot。


定义：

\[
S=
\{s_1,s_2,...,s_n\}
\]


每个：

\[
s_i=(k_i,v_i,m_i)
\]


其中：

k:

key


v:

value


m:

metadata


---

# 9. Memory Manager


负责：

## Add


加入：

\[
W'=W\cup New
\]


---

## Score


计算：

\[
U_i=
\alpha R_i
+
\beta A_i
+
\gamma N_i
-
\delta D_i
\]


其中：

R:

relevance


A:

attention usage


N:

novelty


D:

redundancy


---

## Evict


超过 budget：

\[
|W|>B
\]


删除：

\[
argmin(U_i)
\]


---

# 10. Thinking Core


定义：

\[
T_\theta
\]


输入：

当前状态：

\[
H_t
\]


Working Memory：

\[
W_t
\]


输出：

\[
H_{t+1}
\]


公式：

\[
H_{t+1}
=
T_\theta(H_t,W_t)
\]


---

# 11. Thinking Loop


最大：

\[
R_{max}
\]


例如：

\[
R_{max}=4
\]


执行：


```python
for r in range(Rmax):

    hidden = thinking_core(
        hidden,
        memory
    )

```


---

# 12. Adaptive Halting Router


每轮预测：

是否停止。


\[
p_s
=
\sigma(W_sH_r)
\]


其中：

\[
p_s=P(stop)
\]


规则：

```python
if p_stop > threshold:

    break

```


---

# 13. Dynamic Retrieval + Thinking


完整流程：


```text

Hidden

 |

 v


Thinking Core

 |

 v


Thinking Router

 |

 +---------+

 |         |

stop    continue


          |

          v


 Retrieval Router


          |

   +------+------+

   |             |

reuse       retrieve


                 |

                 v


          Update Memory


                 |

                 v


          Next Loop


```


---

# 14. Working Memory Attention


重点：

Thinking Core 不直接 Attention 1M。


而：

\[
Attention(Q,H_W)
\]


其中：

\[
H_W
\]

是 Working Memory。


复杂度：

\[
O(S_W^2)
\]


而不是：

\[
O(S_L^2)
\]


---

# 15. Memory KV Cache


Working Memory 编码：

\[
K_W,V_W
\]


缓存。


Loop:

\[
Q_rK_W^T
\]


复用：

\[
K_W,V_W
\]


减少重复计算。


---

# 16. Coda


输出：

\[
Y=C(H)
\]


负责：

- token prediction
- output formatting


---

# 17. 完整 Forward


PyTorch 伪代码：

```python
def forward(input_ids):


    # Prelude

    h = self.prelude(
        input_ids
    )


    # Initial Retrieval

    memory = self.retrieve(
        h,
        long_context
    )


    for r in range(
        self.max_loop
    ):


        # Thinking

        h = self.thinking_core(
            h,
            memory
        )


        # Halt

        stop_prob = self.halt_router(
            h
        )


        if stop_prob > threshold:

            break



        # Retrieval Decision

        retrieve_prob = (
            self.retrieve_router(h)
        )


        if retrieve_prob > threshold:


            memory = self.retrieve(
                h,
                long_context
            )



    output = self.coda(h)


    return output

```


---

# 18. Loss Design


总 Loss:


\[
L=
L_{task}
+
\lambda_1L_{halt}
+
\lambda_2L_{retrieve}
+
\lambda_3C_{memory}
+
\lambda_4C_{compute}
\]


---

# 19. Task Loss


标准：

\[
L_{task}
=
CE(y,\hat y)
\]


---

# 20. Halting Loss


监督：

continue/stop


\[
L_h
=
BCE(p_s,y_s)
\]


---

# 21. Retrieval Loss


监督：

是否需要 Retrieval。


\[
L_r
=
BCE(p_r,y_r)
\]


---

# 22. Compute Regularization


Memory:

\[
C_m=
\frac{|W|}
{B_{max}}
\]


Thinking:

\[
C_t=
\frac{R}
{R_{max}}
\]


总：

\[
C=
C_m+C_t
\]


---

# 23. PyTorch Module Design


目录：

```
minimind_x/

model/

 ├── prelude.py

 ├── thinking_core.py

 ├── coda.py


 ├── retrieval/

 │    ├── qsa_retriever.py

 │    ├── indexer.py

 │    └── selector.py


 ├── memory/

 │    ├── working_memory.py

 │    ├── memory_manager.py

 │    └── memory_cache.py


 ├── router/

 │    ├── halt_router.py

 │    ├── retrieval_router.py

 │    └── budget_router.py


 └── minimind_x.py

```


---

# 24. 核心 Class


## MiniMindX


```python
class MiniMindX(nn.Module):

    def __init__(self):

        self.prelude

        self.retriever

        self.memory

        self.thinking_core

        self.halt_router

        self.retrieve_router

        self.coda


```


---

## Retriever


```python
class SparseRetriever(nn.Module):


    def forward(
        self,
        query,
        context
    ):

        candidates = self.coarse_index(
            query
        )

        blocks = self.selector(
            candidates
        )

        return blocks

```


---

## Working Memory


```python
class WorkingMemory:


    def add(
        self,
        blocks
    ):


    def evict(
        self
    ):


    def get_kv(
        self
    ):

```


---

## Thinking Core


```python
class ThinkingCore(nn.Module):


    def forward(
        self,
        hidden,
        memory_kv
    ):

        hidden = layer4(
            hidden,
            memory_kv
        )

        hidden = layer5(
            hidden,
            memory_kv
        )


        hidden = layer6(
            hidden,
            memory_kv
        )


        return hidden

```


---

# 25. 训练阶段


## Pretrain


目标：

学习：

- context encoding
- retrieval ability
- stable loop


---

## SFT


增加：

- Long QA
- Code Agent
- Multi-hop
- Planning


---

## RL


Reward:


\[
Reward
=
Correctness
-
\lambda MemoryCost
-
\lambda LoopCost
\]


---

# 26. 推理模式


## Fast Mode


限制：

```text
Memory <=2K

Loop <=1

```


---

## Normal Mode


```text
Memory=4K

Loop=2

```


---

## Deep Mode


```text
Memory=16K

Loop=4

```


---

# 27. 最终架构总结


MiniMind-X:

\[
\boxed{
LongContext
+
SparseRetrieval
+
AdaptiveMemory
+
AdaptiveReasoning
}
\]


模型拥有：

## Memory Controller

决定：

我要看多少。


## Retrieval Controller

决定：

我要找什么。


## Thinking Controller

决定：

我要想多久。


---

# 28. 最终研究目标


验证：

> 一个语言模型是否可以自主学习最优计算路径。


即：

\[
\boxed{
Read Less
\rightarrow
Think Better
\rightarrow
Compute Efficiently
}
\]


MiniMind-X 不追求最大参数。


追求：

# Adaptive Intelligence

让模型像一个会管理自己计算资源的智能体。