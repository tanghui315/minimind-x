# MiniMind-X 技术审查

2026-11-24 · 审查版本 `0f1b4f0` · 本地 CPU / PyTorch 2.3.1 / Transformers 5.5.3

**判断：保留中段共享计算和独立证据 Reader，先修训练与评测，再决定是否扩大模型。** 项目已经可以研究一个有价值的问题：给定有限的延迟预算，一道题应该再算一轮，还是再读一份证据？当前代码分别实现了循环、检索、记忆和路由，但训练时路由看到的状态、部署时实际执行的动作以及评测时加载的模型还没有完全对齐。现有结果支持继续做受控实验，尚不足以支持 V2 的质量、百万 token 问答能力或端到端成本优势。

## 1. 文档所处的层次与实际执行图

| 仓库内容 | 它能回答的问题 | 审查判断 |
| --- | --- | --- |
| 中英文 README、原版模型、原版训练器和 API | MiniMind 基座具备哪些训练与推理入口 | Dense/MoE、预训练、SFT、LoRA、蒸馏、偏好训练、Agent 和服务构成原版生态。README 的发布指标与 API 示例不能直接归到 X 版 |
| 自适应潜变量、终局架构、数学定义三份设计 | 要把计算用在哪里 | 固定 Prelude/Core/Coda 分区，中段共享权重，外部文档分块，Router 控制深度与检索；区域本身默认固定 |
| 数据策略、三阶段 Recipe、Scaling Law 方案 | 应如何训练并证明收益 | 给出了数据族、预算 oracle、分阶段训练和对照矩阵。具体脚本对阶段继承、预算匹配和数据切分的实现见下文 |
| 编码任务清单、ARCHITECTURE | V1 怎么实现，记录过哪些实验 | 9 月的 zone 对比和多出口损失属于早期循环主干。不能用于推断随后增加的 V2 Reader、MHAR 或自适应策略的贡献 |
| V2 技术方案、V2 残差任务清单 | 哪些机制增补了 V1，哪些已有实测 | 从设计稿到实现有时间差。当前源码已经包含三项 V2 模块，配置默认仍为 V1。任务清单最后记录了 Stage‑0 DDP 故障和修复，未附可复查的完整 V2 大规模训练及能力报告 |

以 V2 的带上下文问答为例：

1. **外部文档**分成块。块内 token embedding 均值经过线性投影形成粗索引。问题经 Prelude 和 query 投影，先按余弦相似度选候选，再用候选块的 Prelude token 表示做最大相似度重排。粗检漏掉的块，本轮精排无法找回。
2. **工作记忆**按源块编号去重，在预算内保留证据。驱逐分数使用相关性、使用情况和冗余等启发式特征。当前默认权重固定，usage 的输入是选中块的命中标志；重复命中、Reader 真正使用和证据对答案有贡献是三件可分别测量的事。novelty 在多块时等于 1 减 redundancy，两项在固定预算排序时实际共用一个自由度。
3. **问题和已生成 token**进入主干。设 Prelude 输出为 E，每轮先由 Loop‑MHAR 按特征子空间从 E 与历轮已提交状态中读取，再经共享 Core。Core 首块的独立 Reader 读取外部证据并门控注入；更新门控制从上一状态向本轮候选移动的幅度。Coda 产生输出。V1 则将记忆 K/V 接入 Core 注意力，两条路径在计算和权重上都不同。
4. **控制与缓存**分开理解。halt 管停机，retrieval 在复用、缓存候选重排、全局刷新之间选择，budget 管容量。初次检索是强制动作。路由并不直接学习逐块驱逐。主干历史 K/V 按循环深度分槽；固定 R 共享参数仍需执行 R 次计算，并占用相应缓存。带版本的外部记忆、当前 token 的循环历史、历史 token 的 K/V 分属三个生命周期。

源码入口：[主模型](../model/minimind_x/minimind_x.py) · [循环核心](../model/minimind_x/thinking_core.py) · [控制器](../model/minimind_x/controller.py) · [两级检索](../model/minimind_x/retrieval/qsa_retriever.py) · [记忆管理](../model/minimind_x/memory/memory_manager.py) · [原版服务](../scripts/serve_openai_api.py)

## 2. 本次验证了什么

原有测试 **140/140 通过**，包括 R=1 等价、按轮 K/V、空记忆、部分因果约束、参考执行器对齐及小任务过拟合。另做了以下独立实验。除 tokenizer 和源码外，这些实验使用临时合成输入和小模型；分布式实验使用两个独立 CPU 进程与 Gloo。所有参数数量差异均指 state_dict 中的命名条目。

| 实验 | 本次观测 | 解释 |
| --- | --- | --- |
| 双进程运行真实 SFT‑C step、深度 sweep 和 GRPO rollout 函数 | 两个 rank 均重现 query_proj.weight 被标为 ready 两次、DDP 缺少 config、DDP 缺少 generate | 是三个能分别触发的入口问题。Stage‑0 的 DDP 修复没有覆盖这些路径 |
| 同一问题换答案后缀，给统计旁路非零权重 | 前 4 个 token 的第一轮 committed hidden 最大差为 0；三轮 halt logit 绝对差分别为 0.04039、0.02515、0.02387 | 隐状态的前缀因果性成立，halt 统计汇聚仍带入未来信息。此实验构造了可达参数状态，不表示已有权重具有同等泄漏幅度 |
| 用项目保存器写小型 V2 权重，按评测入口构造 V1 并宽松加载 | 0 个缺失项、23 个额外项，其中 22 个来自 V2 模块，另一个是元数据 | 仅检查“能加载”会误判；同形主干能运行，但评测对象已改变 |
| 调用带上下文 CLI 使用的参数组合 | 生成前出现 return_stats 重复传参的 TypeError | 修复加载器后，带文档的命令仍需独立端到端验证 |
| 64 类等概率的预测器，运行 eval_ppl | 输出 4.15888，而 perplexity 为 64 | 当前 JSON 的 ppl 字段存的是平均 token NLL |
| 默认生成方式的代表性 Needle，指定 8,192 token，深度 0.1 / 0.5 / 0.9 | 实际为 6,495 / 6,491 / 6,468 token。标注块为 3 / 16 / 28，用仓库 tokenizer 的字符 offset 算得为 2 / 12 / 22 | 三条样本均发生监督错位。这是可复查反例，不能当作整个数据集的错误率 |
| 一条真实 token，块长 8；另单独改变 embedding 重建索引 | Reader 标记 8 个位置均有效；粗 key 已变化，manifest 中 key_encoder_hash 相同 | 尾块填充影响检索及容量统计；指纹没有覆盖完整 key 生成过程 |
| 两条训练、两条验证记录组成的混合文件；短文长序列训练 | SFT‑C 相同数据构造参数读到 4 条，显式 train 得 2 条；新证据时间字段在样本中消失。一条短文设 8,192 长度，仅 9 个非忽略标签 | 分割与证据时间线需要在训练入口执行。张量长度与有效训练长度应分开记录 |
| **精确 1,000,000 个随机 token**，D=32，块长 256，4 个粗检候选，R=2 | 成功建 3,907 块索引，生成 2 个 token，终态工作记忆容量为 1,024 | 证实外部库结构可以容纳这个长度。小模型、随机资料和固定深度不提供任务准确率、真实权重表现或 GPU 延迟证据 |

复核环境与项目 requirements 的版本组合有差异。系统 pytest 在导入本机 readline 时崩溃，禁用该导入和第三方 pytest 插件后完整运行原测试；Gloo 双进程使用文件汇合。未运行 NCCL、S/M/L 长训或用历史 H200 权重复测。当前检出仅见两份文件名带 64 的本地 checkpoint，未见支撑 9 月表格的逐样本记录、完整对照权重和多种子日志。

## 3. 应在下一轮大训练前处理的问题

### P0 · 训练和评测可以直接失败，或得出错误结论

**分布式可训练路径跨越了 DDP 的 forward 边界。** SFT‑C 的主前向经过 DDP，检索 CE / 蒸馏却在返回后直接用 raw retriever 重算。find_unused_parameters 对该分支做出的判断与随后反传冲突。另有 oracle 直接读包装对象的 config，GRPO 对包装对象调用 generate。建议把全部可微分支放进同一次被包装的前向，纯采样和无梯度 oracle 明确使用底层模型。验收应让两个 rank 使用不同样本，跑过有标签、空库及带蒸馏的多次优化更新，核对梯度同步和 checkpoint。 [SFT-C](../trainer/train_sft_x.py) · [控制器训练前向](../model/minimind_x/controller.py) · [oracle](../trainer/oracle_labels.py) · [GRPO](../trainer/train_grpo_x.py)

**V2 的 halt 监督有一条隐蔽的未来信息通道。** 主干把供路由池化的 hidden 截到 query_len，但 ThinkingCore 的 gate_mean、delta_cand、delta_commit 已经沿完整 teacher-forcing 序列求均值。detach 只阻断梯度，数值依然包含后缀。统计投影从零初始化使初始模型通过某些替换后缀测试。应在归约前应用可见前缀与 padding mask，回归时让旁路非零，并同时断言 halt logits、动作与可见状态一致。 [统计计算](../model/minimind_x/thinking_core.py) · [停机头](../model/minimind_x/router/halt_router.py) · [路由汇总](../model/minimind_x/minimind_x.py)

**训练、评测和服务对模型身份的处理不同。** 训练加载器会检查 V2；eval_x、eval_suite、eval_needle、eval_depth_dist 各自构造默认配置并宽松加载。部分入口的 hidden_size、层数有具体默认值，还可能覆盖 preset。eval_x 的 context_file 分支同时由调用方与被调用方指定 return_stats。原版内置 API 的本地 pth 分支实例化的是 MiniMindForCausalLM。建议统一依据 checkpoint 保存的完整模型配置构建实例，显式区分迁移和同架构加载，再以保存前后同一输入的 logits、记忆轨迹及两步生成验收。 [训练加载器](../trainer/x_utils.py) · [推理 CLI](../scripts/eval_x.py) · [评测套件](../scripts/eval_suite.py) · [服务入口](../scripts/serve_openai_api.py)

**数据真值在生成器、tokenizer、训练器之间发生变化。** 七族生成器把 train/val 写入同一个文件，SFT‑C 没传 split；当直接使用这种文件时，验证记录参加训练。new_evidence 写了可见轮次，dataset 丢弃该字段，初始检索可搜索全库。通用 Needle 生成器按字符估算长度和块号，未输出供训练器重算的 gold_span。修复应以实际 token offsets 为准，给每个证据块保存来源、可见时点和有效长度，并把过滤应用到候选选择之前。训练、验证应按文档或事实组隔离，统计真实交集。 [七族生成器](../scripts/data_gen/gen_evidence_families.py) · [Needle](../scripts/data_gen/gen_needle.py) · [检索数据集](../dataset/retrieval_qa_dataset.py) · [SFT-C 数据入口](../trainer/train_sft_x.py)

### P1 · 会削弱训练信号，或使架构对比失真

| 路径 | 当前行为与影响 | 建议 |
| --- | --- | --- |
| [三阶段启动](../scripts/launch_h20_8card.sh) | SFT-A/B/C 默认都读 pretrain_x；GRPO 三相默认都读 sft_x_c，且共用输出前缀。H20 pretrain 分支与单独的 run_stage0.sh 也选择不同配置。直接按文档依次运行无法保证阶段继承 | 让每阶段显式记录输入 checkpoint 身份、输出和实际配置。用缩小模型跑完一次全链，核对父子权重。SFT 在累积窗口尾部提前退出还可能保存最后一次更新以前的参数 |
| [长程训练](../trainer/train_long_context.py) | 普通数据分支随机取单篇文档后补齐；分布式 loader 没有 DistributedSampler，dataset 又以相同种子持有私有 RNG 且忽略 index，造成跨 rank 或 worker 的重复采样风险。默认 SFT-C 外部 context 也被主干位置长度上限截断 | 按有效 token 打包及分片，统计训练长度分布、重复率、答案证据截断率。分开配置外部库容量和主干位置长度。[数据实现](../dataset/long_context_dataset.py) |
| [固定 R GRPO](../trainer/train_grpo_x.py) | rollout 使用裸文本问题和 model.generate，没有 controller、context 或 memory_provider。正常执行时检索与记忆成本均为零；R 由脚本指定。采样还使用温度及 top-k/top-p，随后从原始全词表 logits 重算 logp。可用于研究不同固定深度下的 token 策略，不能据此认定联合动作获得了策略梯度 | 明确定义行为分布和目标策略，保存真实 token 及动作 logp。先对每条轨迹验重放，再给控制器做策略优化。仅检查平均深度是否为整数也会漏掉深度 1、3 交替的情形 |
| [Oracle 与训练时序](../trainer/oracle_labels.py) | SFT-C 的深度 sweep 未传任务外部记忆；三动作 oracle 在 R 个出口上产标签，但最后一次出口之后没有下一轮执行动作。旧二值回退得到的 1 也会在 v2 被解释为 warm。训练整段 teacher forcing，部署逐 token 追加且中途记忆可换版 | 在相同可见前缀、记忆快照和实际可执行动作下生成标签；监督 R−1 个轮间动作。让分段参考执行覆盖真实训练样本，检查逐 token logp。另报 unresolved 比例与全错题的预算行为。[执行器](../trainer/dual_executor.py) |
| [索引一致性](../model/minimind_x/retrieval/block_store.py) | 冻结了 key_proj，但计算 key 还用可训练的 embedding。hash 只覆盖投影；普通 begin 没有填全 tokenizer / model 标识。SFT-C 每步新建 controller，周期分支读 args.step_global，而训练循环更新局部变量，随后 forward 又 begin。当前刷新消融在同一静态模型上重建 | 先确定每请求即时建库还是跨模型版本复用索引。长驻索引可绑定完整 key 编码器快照和 tokenizer 版本，独立测索引漂移、重建时间及缓存失效。[刷新消融](../scripts/exp_key_refresh_ablation.py) |
| [尾块与来源](../model/minimind_x/memory/working_memory.py) | 粗排均值、精排 max-sim 和 V2 Reader 目前可纳入尾块补零位置。actual_valid_tokens 使用槽容量；unique_blocks 统计最近检索的集合，与驱逐后的常驻集合可能不同 | 从切块开始传 token 级 mask，分别记录库中原始 token、保留有效 token、已分配容量、最终证据 ID。强化学习成本应对应真实开销 |
| [成本模型](../model/minimind_x/flops.py) | total 按整条序列累计，eval_suite 记作 flops_tok。V2 Reader 只在 Core 首块执行，估算却乘 Core 层数，并与旧 memory attention 并计；无记忆时仍计部分 Reader 投影。一些门控项又漏乘 token 数。Model C 的稀疏掩码和 local 路径依然物化 dense scores | 把索引、候选编码、跨轮读取、主干 prefill、decode 分项；核对 batch、token 和触发次数，再用 profiler、峰值显存、TTFT、TPOT 校准。[稀疏实现](../model/minimind_x/retrieval/sparse_attn.py) |
| [能力评测](../scripts/eval_needle.py) | needle 使用答案子串匹配；多针网格每个单元都传相同深度列表，数据文件默认只看开头几个样本。backbone 超长时从拼接文本右端裁剪，可能丢掉问题。深度评测的 acc_fixed 是任意固定 R 答对即记分，stop_accuracy 用生成过程平均 R 比较单一 oracle R | 单独报告每档固定 R、可解题比例、oracle 上界、路由实际准确率及按 token 决策质量。按真实长度、证据深度、任务族和分母分层，保留问题与生成空间。[深度评测](../scripts/eval_depth_dist.py) |

成本之外还有一项训练信号值得关注：budget router 在 SFT-C 默认参数中关闭监督，但控制器仍用其 argmax 决定实际预算。若起点是缺少已训预算头的 checkpoint，初期预算来自无监督的初始化。宜先固定预算获得稳定基线，再报告开启预算学习后的质量和成本。[训练开关](../trainer/train_sft_x.py) · [预算执行](../model/minimind_x/controller.py)

### 物理开销如何随模型与资料长度变化

设外部资料为 N 个 token，块长 b，粗检候选 k 个块，主干工作序列为 S 个 token，工作记忆为 M 个 token。粗排每次扫描约 N/b 个向量；首次精排最多需要对 k 个候选块运行 Prelude，再在候选 token 上求相似度。缓存命中后的重排省去这些候选的再编码。Reader 的注意力匹配规模约为每轮 S×M，Core 自注意力在完整 prefill 时则随 S² 增长。这解释了为何把 N 放到外部能改变可处理长度，也解释了不同请求是否复用同一文档会显著改变端到端成本。当前 generate_with_context 每次调用都 begin 建库，隐状态缓存的上限也应以字节计入冷、热请求的测量。[建库与候选缓存](../model/minimind_x/retrieval/block_store.py) · [生成入口](../model/minimind_x/controller.py)

深度自适应还有独立的系统约束。令 P、C、T 分别表示 Prelude、Coda、Core 层数，单样本选 R 轮时，主干执行层数约为 P+C+T×R。解码缓存却在每个 Core 层分配 R_max 个槽，浅停 token 的末轮 K/V 会填入更深的槽。在这套实现中，序列缓存的元素量近似与 P+C+T×R_max 成正比。混合 batch 又会一直算到整批都停，浅样本的平均选中轮数无法直接换算为节省的毫秒。先比较 batch=1 和混合 batch，再决定是否引入 MoR 式 active-token 压缩、KV 别名或分桶调度。[缓存写入](../model/minimind_x/minimind_x.py) · [批内早停](../model/minimind_x/thinking_core.py)

## 4. 已有实验证据能支撑哪一步决策

仓库 9 月记录的 2,000 步损失为 Dense R1 4.4623、中段 R2 4.5419；8,000 步时分别为 3.1343、3.1290，差值缩至 0.0053 并换向。单次结果值得复验。当前 zone runner 从同一个 ds 随机采训练和验证样本，没有固定互斥验证集；相同优化步数也没有对齐总算量。按 runner 的默认 8 层配置与当前估算器，以中段 R2 为 1，Dense、全层 R2、晚中段 R3 约为 0.727、1.455、1.091。该比例是对当前脚本的核算，历史作业的具体命令和硬件利用率无法由表格还原。[原实验表](./MiniMind-X%20编码任务清单.md) · [当前实验入口](../scripts/run_zone_experiments.py)

多出口损失约为 1.94、1.90、1.89、1.90，说明这个训练记录中第 3 轮的**平均**损失最低。平均曲线无法区分“所有题都适合 R3”和“不同题适合不同轮，平均后 R3 最好”。后者才提供学习自适应深度的空间。应在同一保留样本上保存各轮损失和生成正确性，估计逐样本 oracle 相对最佳固定深度的上界，再测 Router 能捕获其中多少。若上界很小，即使把 Router 训到完美，工程收益也有限。[任务清单](./MiniMind-X%20编码任务清单.md)

矩阵脚本写命令、运行时长和 stdout/stderr 尾部；拟合脚本需要带 N、D、R、M 和验证损失的逐行记录，两者尚缺直接数据链。小样本共线时，普通最小二乘仍可输出指数与 R²。先分开控制参数量、有效训练 token、循环深度与读入记忆量，报告设计矩阵秩和跨种子区间，再解释 scaling exponent。[矩阵](../scripts/run_experiment_matrix.py) · [拟合](../scripts/scaling_fit.py)

## 5. 值得结合的前沿研究

这里按对 MiniMind-X 的预期信息价值排序。论文中的模型规模、训练数据和加速数字都属于原论文；下列均为待测实验。

| 方向与论文 | 在本项目怎样落地 | 判定是否值得保留 |
| --- | --- | --- |
| **首选：学习再算或再读的边际收益。** Huginn 验证了共享深度可以承载隐式计算；MoR 给出按 token 分配递归深度和有选择地缓存 K/V 的实例 | 先固定工作记忆容量，在同一状态分叉“现在停止、再算一轮复用、再算并重排、再算并全局刷新”。用后续答案损失改善除以实测增量毫秒，训练小型动作价值头；短视分支漏掉必须两跳才见效的样本时，另测两步回报 | 与最佳固定 R、单次静态检索、原三头策略做配对比较。报告同准确率的 GPU 毫秒、同延迟的正确率、误停止率，以及 oracle 到实际策略的差距。[Huginn](https://arxiv.org/abs/2502.05171) · [MoR](https://arxiv.org/abs/2507.10524) |
| **检验循环为何饱和。** 2026 年 STARS 针对加深循环后质量回落，引入随机深度和局部动力学稳定性正则；Dreamer 的 11 月修订讨论深度注意力与多重预算匹配 | 先在训练范围 R=1…4 和范围外 R=5…8 观察逐轮答案修正、正确变错误、状态增量、历史读取熵。按“共享 Core → 加更新门 → 加历史读取 → 两者一起”消融。若状态发散再试低频稳定性正则；若很早收敛却持续答错，优先检查内容和证据 | 看难题正确率、正确变错误率和真实总开销。把状态变化小单独作为停机准则会混淆“已解出”和“停滞”。现有 Loop-MHAR 沿整轮 committed state 读取，与按层残差读取的设计粒度不同。[STARS](https://arxiv.org/abs/2605.26733) · [Dreamer](https://arxiv.org/abs/2601.21582) · [Multi-Head Attention Residuals](https://arxiv.org/abs/2607.27230) · [Gated Recurrent Transformers](https://arxiv.org/abs/2608.15062) |
| **让第二次检索能提出不同的问题。** RLM 将长文档作为可反复访问的外部环境 | 为仓库问答构造必须跨文件接力的题。第一跳读出函数、设备或版本标识，第二跳以该标识和当前问题生成检索 query。先比较固定两跳、BM25 加向量候选、当前 hidden 重排；必要时再引入可学习的查询改写。保留证据 ID，允许缺证拒答 | 同库、同总读入 token、同延迟预算，分报两跳证据齐全率、给定正确证据后的答案准确率、错版本采信率和无证据拒答率。这样能定位损失发生在搜索、读取还是推理。[Recursive Language Models](https://arxiv.org/abs/2512.24601) · [RULER](https://arxiv.org/abs/2404.06654) · [LongBench v2](https://arxiv.org/abs/2412.15204) |
| **条件性投入：按循环轮次分配精度。** 2026 年 10 月的 RAMP 把低精度用于早轮、高精度用于末轮 | 仅在 profile 显示长 prompt 的循环 prefill 是瓶颈时尝试。先固定 R、保持 Reader 和 Router 精度，比较 BF16 与早轮低精度；另测路由决策翻转 | 报端到端首 token 延迟和峰值显存。RAMP 的实验覆盖 H100 与最长 4K 上下文；MiniMind-X 把长资料放在外部，短问题下建库或候选编码可能更贵。[RAMP](https://arxiv.org/abs/2610.13041) |

## 6. 推荐实验顺序与停止条件

1. **恢复一条可信的训练—加载—生成链。** 两个 rank 跑真实 SFT-C 多步；推理载回同一权重；运行带文档 CLI；同前缀换答案、换尚不可见证据时，已发生的决策保持一致。记录每阶段数据、权重与真实优化步数。以同种子同输入的 token、路由和 memory_version 作验收。
2. **先测上界，决定投资对象。** 同题比较四组：只给问题、直接给 gold 证据、单次检索、动态检索。gold 证据也读不懂则优先做 Reader 与答案监督；oracle 检索明显胜出则优先做召回与重排；最佳固定 R 已接近逐题 oracle 则压低动态深度投入。数据按文档、事实、模板或仓库去重，保留一组公开任务与一组独立内部样本。
3. **做能决定是否上 M 的公平实验。** 在小规模固定 tokenizer 和数据顺序，分别做等训练 token、等训练 FLOPs 对照；给推理按相同 P95 延迟及相同准确率画两张曲线。至少三种子报告离散程度，对问题级准确率差做配对 bootstrap。分别记录首次建库和库复用的成本、TTFT、TPOT、实际执行轮数与峰值显存。比较 Dense、固定中段、固定中段加静态检索、动态 V2。
4. **最后验证策略优化。** 在同一轨迹记录每个 token、轮次、动作、预算、证据 ID、版本和行为 logp，逐 token 对齐 rollout 与可微重放，再比较监督动作价值头与联合 RL。若较简单的固定预算或静态检索已经占据相同质量—延迟前沿，优先采用简单模型。完成一次真实多跳任务上的可重复优势，比提前拟合 1.5B scaling 曲线更能决定项目去向。
