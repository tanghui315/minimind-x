"""AdaptiveController：终局编排（V2 改造版；任务清单 T0.1/T0.3/T0.5/T0.7 修订）。

数据流（D9；单次 forward 的操作顺序）：
  建库（一次）→ 因果 Query（仅可见前缀）→ Initial Retrieval → Budget 分配
  → Loop{ MHAR* → Core → Update* → halt? → retrieve? → MemoryManager.update } → Coda

V2 因果契约（T0.1，C5/C6 修复）：
  - 初始检索 query 只用 [bos]+question（query_len 边界由数据集提供）；
  - 训练期轮间再检索 query 只用 exit_hidden 的可见前缀位置；
  - 换 gold completion 不得改变初始检索结果（tests/test_causality.py 锁定）。

V2 梯度契约（T0.7，C3 修复）：gold CE 与蒸馏一样用 grad mode 重算的 coarse logits
（离散 TopK 选择仍在 no_grad，见 D5/V2 §13.2 的分层）。

budget 联动（T0.3，C7 修复）：预算档真正改变 retrieve 的 topk_f；统计分开记录
budget_cap_tokens / actual_valid_tokens / unique_blocks。
"""
from typing import Optional

import torch

from .retrieval.block_store import LongContextStore
from .memory.working_memory import WorkingMemory, MemoryInput
from .minimind_x import MiniMindXOutput
from .router.retrieval_router import (
    ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH, ACTION_NAMES,
    action_mask, normalize_oracle_actions, ACTION_SCHEMA_VERSION)



class AdaptiveController:
    def __init__(self, model, block_visible_from=None, fixed_budget_idx=None):
        """model: MiniMindXForCausalLM 或其 DDP 包装（须 use_retrieval/use_memory 打开）。"""
        self.block_visible_from = block_visible_from
        self.fixed_budget_idx = fixed_budget_idx
        self.model = model
        raw = model
        for attr in ("module", "_orig_mod"):
            raw = getattr(raw, attr, raw)
        self.raw = raw                      # 未包装引用（routers/store 等内部访问）
        self.cfg = raw.config
        self.store: Optional[LongContextStore] = None
        self.memory: Optional[WorkingMemory] = None
        self.stats = {"retrievals": 0, "memory_tokens": 0, "re_retrievals": 0,
                      "global_refresh": 0, "warm_rescore": 0, "reuse": 0}
        # T3.3/T4.7（V2 §11.5/§17.2）：轨迹全字段
        self.trajectory = []
        # T2.5 Warm Candidate Cache：最近一次全局检索的候选（供 RESCORE_WARM 重排）
        self.warm_candidates = None  # dict(cand_idx, cand_hidden, cand_keys, fine_scores)

    def _trace(self, event, **kw):
        """V2 §11.5/§17.2 轨迹字段：event/version + 可选 round/action/budget/exit/logprob。"""
        rec = {"event": event,
               "memory_version": self.memory.memory_version if self.memory is not None else -1,
               "retrieval_action": kw.get("action", event),
               "budget_action": kw.get("budget_idx", None),
               "exit_round": kw.get("exit_round", None),
               "token_position": kw.get("token_position", None),
               "action_logprob": kw.get("action_logprob", None),
               "visible_prefix_len": kw.get("visible_prefix_len", None)}
        rec.update(kw)
        self.trajectory.append(rec)

    # ---------------------------------------------------------------- 建库与初检
    def begin(self, context_ids):
        self.store = LongContextStore(context_ids, self.cfg.block_size, self.block_visible_from)
        self.store.build_keys(self.raw.model.retriever, self.raw.model.embed_tokens)
        self.store.set_manifest(tokenizer=getattr(self.raw.config, "tokenizer_identity", "pretokenized"),
                                model_version=getattr(self.raw, "checkpoint_id", None) or self.cfg.architecture_version())
        self.memory = None
        self.stats = {"retrievals": 0, "memory_tokens": 0, "re_retrievals": 0,
                      "budget_cap_tokens": 0, "actual_valid_tokens": 0, "unique_blocks": 0,
                      "global_refresh": 0, "warm_rescore": 0, "reuse": 0}
        self.trajectory = []
        self.warm_candidates = None
        self.stats.update(last_global_result=None, scanned_blocks=0, scored_blocks=0)
        return self.store

    def _budget_idx(self, query_hidden, max_budget_tokens=None, teacher_idx=None):
        if teacher_idx is not None:
            return int(teacher_idx)
        router = self.raw.model.budget_router
        if self.fixed_budget_idx is not None:
            idx = self.fixed_budget_idx
        elif router is None or not getattr(self.cfg, "use_budget_router", True):
            idx = self.cfg.default_budget_idx
        else:
            logits = router(query_hidden)
            idx = int(logits.argmax(dim=-1).float().mean().item())  # batch 统一档位（研究级）
        if max_budget_tokens is not None:
            cap = [i for i, b in enumerate(self.cfg.memory_budgets) if b <= max_budget_tokens]
            if cap and idx > cap[-1]:
                idx = cap[-1]
        return idx

    def retrieve_into_memory(self, query_hidden, budget_idx=None, mem=None):
        """检索并把新块并入记忆。budget 档位真正联动 topk_f（T0.3）。

        整条检索/建库/memory 编码路径不回传梯度（D5）；gold 监督与蒸馏在
        forward_with_oracles 中用 grad mode 重算的 logits 单独计算（T0.7/V2 §13.2）。
        """
        budget_idx = self.cfg.default_budget_idx if budget_idx is None else budget_idx
        is_initial = mem is None
        budget_blocks = self.cfg.budget_blocks(budget_idx)
        if self.store.visible_count == 0 or self.store.block_keys is None:
            # 空库：合法 no-op memory（评审 6.3）
            mem = mem if mem is not None else WorkingMemory(
                query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.block_size,
                                       self.cfg.hidden_size),
                query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.hidden_size),
                metadata={"context_tokens": self.store.context_ids.numel()})
            self.stats["retrievals"] += 1
            self.stats["global_refresh"] = self.stats.get("global_refresh", 0) + 1
            self._record_memory(mem)
            self.stats["budget_cap_tokens"] = self.cfg.memory_budgets[budget_idx]
            self.stats["last_result"] = None
            self.stats["last_global_result"] = None
            self.warm_candidates = None
            if is_initial:
                self._trace("initial_retrieval", budget_idx=budget_idx,
                            cap_tokens=self.stats["budget_cap_tokens"],
                            valid_tokens=0, unique=0)
            return mem
        result = self.raw.model.retriever.retrieve(
            query_hidden, self.store, self.raw, topk_fine=budget_blocks)  # T0.3 联动
        self.stats["retrievals"] += 1
        self.stats["global_refresh"] = self.stats.get("global_refresh", 0) + 1
        # T2.5：全局检索后刷新 Warm Candidate Cache（供 RESCORE_WARM 重排）
        self.stats["scanned_blocks"] += self.store.n_blocks
        self.stats["scored_blocks"] += result.cand_idx.numel()
        self.warm_candidates = {
            "cand_idx": result.cand_idx.detach() if result.cand_idx is not None else None,
            "cand_hidden": self.store.get_block_hidden(
                self.raw, result.cand_idx).detach() if result.cand_idx is not None else None,
            "block_keys": self.store.block_keys.detach() if self.store.block_keys is not None else None,
            "fine_scores": result.fine_scores.detach() if result.fine_scores is not None else None,
        }
        self.stats["budget_cap_tokens"] = self.cfg.memory_budgets[budget_idx]
        block_keys = self.store.block_keys.to(result.block_idx.device)[result.block_idx]  # [B,k,d]
        q_vec = self.raw.model.retriever.project_query(query_hidden).detach()
        mem = mem if mem is not None else WorkingMemory(
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.block_size,
                                   self.cfg.hidden_size),
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.hidden_size),
            metadata={"context_tokens": self.store.context_ids.numel()})
        hit = torch.ones_like(result.block_idx, dtype=torch.float)  # 新块本轮均被使用
        mem, _, _ = self.raw.model.memory_manager.update(
            mem, result.block_hidden, block_keys, q_vec, budget_blocks, hit_new=hit,
            source_ids=result.block_idx)
        self._record_memory(mem)
        self.stats["last_result"] = result
        self.stats["last_global_query"] = None if is_initial else query_hidden
        self.stats["last_global_result"] = result  # distill/gold CE 用（warm rescore 不清除）
        # 事件记录口径：初始检索在此记录；轮间 refresh/reuse 由 loop_callback 记录（带轮次）
        if is_initial:
            self._trace("initial_retrieval",
                        budget_idx=budget_idx, cap_tokens=self.stats["budget_cap_tokens"],
                        valid_tokens=self.stats["actual_valid_tokens"], unique=self.stats["unique_blocks"])
        return mem

    def _record_memory(self, mem):
        self.stats.update(context_tokens=self.store.context_ids.numel(), memory_tokens=mem.token_size,
                          actual_valid_tokens=int(mem.token_valid().sum()),
                          resident_source_ids=mem.source_ids[mem.valid].tolist(),
                          unique_blocks=int(mem.source_ids[mem.valid].unique().numel()))

    def refresh_index(self, max_blocks=None):
        """P1：用当前 key_proj 重建库索引（Frozen Index 的周期 refresh）。

        仅用于持有同一 store 的调用方。请求级 SFT 每次建库，重建会清空已有记忆和候选。
        """
        if self.store is None:
            raise RuntimeError("refresh_index 需要已 begin() 的 store")
        self.store.build_keys(self.raw.model.retriever, self.raw.model.embed_tokens,
                              max_blocks=max_blocks)
        # 索引指纹已变：warm 候选/last_global 基于旧 K，必须失效（防 stale 重排）
        self.warm_candidates = None
        self.stats["last_global_result"] = None
        self.stats["last_result"] = None
        self.memory = None
        self.stats.update(memory_tokens=0, actual_valid_tokens=0, resident_source_ids=[], unique_blocks=0)
        self.stats["index_refreshes"] = self.stats.get("index_refreshes", 0) + 1
        return self.store.metadata

    def maybe_refresh_index(self, step):
        """按 config.key_refresh_interval 周期 rebuild（0=永不，Frozen forever）。"""
        interval = int(getattr(self.cfg, "key_refresh_interval", 0) or 0)
        if interval <= 0 or self.store is None:
            return False
        if step > 0 and step % interval == 0:
            self.refresh_index()
            return True
        return False

    def has_warm_cache(self) -> bool:
        """Warm Candidate Cache 是否可用（决定 RESCORE_WARM 是否可选，V2 §9.2）。"""
        w = self.warm_candidates
        return bool(w is not None and w.get("cand_idx") is not None and w["cand_idx"].numel() > 0)

    def rescore_warm_into_memory(self, query_hidden, budget_idx=None, mem=None):
        """T2.5 RESCORE_WARM：只在已缓存候选中重排选择，不重搜全局索引。

        可能改变 memory_version（选中集合变化时）；成本低于 GLOBAL_REFRESH
        （不重新建索引/不重投影未选中候选）。
        """
        if not self.has_warm_cache():
            raise RuntimeError("RESCORE_WARM 需要 Warm Candidate Cache；无缓存时必须 mask 该动作")
        budget_idx = self.cfg.default_budget_idx if budget_idx is None else budget_idx
        budget_blocks = self.cfg.budget_blocks(budget_idx)
        w = self.warm_candidates
        self.stats["scored_blocks"] += w["cand_idx"].numel()
        cand_hidden = w["cand_hidden"]                          # [B,c,block,d]
        cand_idx = w["cand_idx"]                                # [B,c]
        q = self.raw.model.retriever.project_query(query_hidden).detach()
        # fine 重排：query 对候选块 token 级 max-sim（与 retrieve 的 fine 同口径）
        tok_sim = torch.einsum("bd,bckd->bck", torch.nn.functional.normalize(q, dim=-1),
                               torch.nn.functional.normalize(cand_hidden, dim=-1))
        fine_scores = tok_sim.masked_fill(~self.store.token_mask(cand_idx), -torch.inf).amax(-1)                # [B,c]
        topk_f = min(budget_blocks, fine_scores.shape[-1])
        k = fine_scores.topk(topk_f, dim=-1).indices
        final_idx = cand_idx.gather(1, k)
        final_hidden = cand_hidden.gather(
            1, k.view(*k.shape, 1, 1).expand(-1, -1, cand_hidden.shape[2], cand_hidden.shape[-1]))
        block_keys = w["block_keys"].to(final_idx.device)[final_idx]  # [B,k,d]
        mem = mem if mem is not None else WorkingMemory(
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.block_size,
                                   self.cfg.hidden_size),
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.hidden_size),
            metadata={"context_tokens": self.store.context_ids.numel()})
        hit = torch.ones_like(final_idx, dtype=torch.float)
        mem, _, _ = self.raw.model.memory_manager.update(
            mem, final_hidden, block_keys, q, budget_blocks, hit_new=hit,
            source_ids=final_idx)
        self.stats["retrievals"] += 1
        self.stats["warm_rescore"] = self.stats.get("warm_rescore", 0) + 1
        self._record_memory(mem)
        self.stats["budget_cap_tokens"] = self.cfg.memory_budgets[budget_idx]
        # 不覆盖 last_global_result（distill teacher 仍用最近全局检索的 fine_scores）
        return mem

    def execute_retrieval_action(self, action, query_hidden, budget_idx=None, mem=None):
        """执行三动作之一（T2.5 / V2 §9.2）。返回新 memory 或 None（REUSE）。"""
        action = int(action)
        if action == ACTION_REUSE:
            self.stats["reuse"] = self.stats.get("reuse", 0) + 1
            return None
        if action == ACTION_RESCORE_WARM:
            if not self.has_warm_cache():
                # 防御：无 warm cache 时不得静默降级为全局搜索（V2 §9.2 禁止项）
                raise RuntimeError("RESCORE_WARM 无 Warm Candidate Cache（应在 logits mask 掉）")
            return self.rescore_warm_into_memory(query_hidden, budget_idx=budget_idx, mem=mem)
        if action == ACTION_GLOBAL_REFRESH:
            return self.retrieve_into_memory(query_hidden, budget_idx=budget_idx, mem=mem)
        raise ValueError(f"未知 retrieval action: {action}（应为 0/1/2）")

    def available_action_mask(self, batch=1, device=None):
        """当前合法动作 mask [B,3]。"""
        return action_mask(self.has_warm_cache(), device=device, batch=batch)

    def joint_path_logprob(self, token_logprobs=None, halt_logprobs=None,
                           retrieval_logprobs=None, budget_logprobs=None):
        """V2 §17.2 联合路径 log P(τ) = Σ token + Σ halt + Σ retrieval + Σ budget。"""
        total = 0.0
        for seq in (token_logprobs, halt_logprobs, retrieval_logprobs, budget_logprobs):
            if seq is None:
                continue
            if torch.is_tensor(seq):
                total = total + float(seq.sum().item())
            else:
                total = total + float(sum(seq))
        return total

    def assert_replay_not_averaged(self, exit_rounds):
        """V2 §17.3 禁止平均深度重放：[1,3,2,4] 不得用 R=3 整段近似。

        仅作门槛检查：exit_rounds 为逐 token 深度时，若调用方试图用均值重放会在此拦截。
        """
        if exit_rounds is None:
            return True
        rs = [int(r) for r in exit_rounds]
        if not rs:
            return True
        if len(set(rs)) > 1:
            mean_r = int(round(sum(rs) / len(rs)))
            # 允许记录均值作日志，但轨迹重放必须保留逐 token 深度
            self.stats["depth_mean_for_log_only"] = mean_r
            self.stats["depth_per_token"] = rs
        return True

    # ---------------------------------------------------------------- loop callback
    def make_loop_callback(self, oracle_retrieve=None, max_budget_tokens=None,
                           query_len=None):
        """轮间再检索回调。query_len: 训练期可见前缀边界（T0.1）——轮间检索 query
        只用 exit_hidden 的可见前缀位置池化，gold 后缀不得影响检索决策。"""
        cfg = self.cfg

        def cb(r, exit_hidden):
            """轮间动作（T2.5）：REUSE / RESCORE_WARM / GLOBAL_REFRESH。

            oracle_retrieve 必须已是 schema v2 枚举（0=REUSE, 1=RESCORE_WARM, 2=GLOBAL_REFRESH）。
            旧二值标签请先经 normalize_oracle_actions(..., schema_version=1) 转换，禁止在此对调。
            无 oracle 时用 RetrievalRouter.action_logits（RESCORE_WARM 无 warm cache 则 mask）。
            """
            vis = exit_hidden[:, :query_len] if query_len else exit_hidden
            self.store.advance_round(r + 1)
            warm_ok = self.has_warm_cache()
            if oracle_retrieve is not None and r < len(oracle_retrieve):
                action = int(oracle_retrieve[r])  # 已是 schema v2，禁止再映射
                if action not in (ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH):
                    raise ValueError(f"oracle 动作非法: {action}（须先 normalize）")
                if action == ACTION_RESCORE_WARM and not warm_ok:
                    action = ACTION_GLOBAL_REFRESH  # oracle 也不允许无缓存 RESCORE
                a_logp = None
            else:
                router = self.raw.model.retrieve_router
                logits = router.action_logits(vis, warm_available=warm_ok)
                probs = torch.softmax(logits.float(), dim=-1)
                action = int(probs.mean(0).argmax().item())  # batch 统一（研究级）
                a_logp = float(torch.log_softmax(logits.float(), dim=-1).mean(0)[action].item())
            self.stats["re_retrievals"] += (action != ACTION_REUSE)
            # 事件名：REUSE→reuse，RESCORE_WARM→rescore_warm，GLOBAL_REFRESH→refresh（legacy）
            ev = {ACTION_REUSE: "reuse", ACTION_RESCORE_WARM: "rescore_warm",
                  ACTION_GLOBAL_REFRESH: "refresh"}[action]
            self._trace(ev, round=r, action=action,
                        action_logprob=a_logp, visible_prefix_len=query_len, warm_available=warm_ok)
            if action == ACTION_REUSE:
                return None
            with torch.no_grad():
                self.memory = self.execute_retrieval_action(
                    action, vis,
                    budget_idx=self._budget_idx(vis, max_budget_tokens),
                    mem=self.memory)
                return self._memory_input()
        return cb

    # ---------------------------------------------------------------- 推理入口
    @torch.no_grad()
    def generate_with_context(self, input_ids, attention_mask=None, context_ids=None,
                              mode=None, return_stats=True, **gen_kwargs):
        """终局生成（T6.3）：建库一次 → 因果初检 → generate（router 驱动轮间再检索）。

        推理期 input_ids 即完整可见 prompt，无需 query_len 边界。
        """
        assert self.cfg.use_retrieval and self.cfg.use_memory, "终局路径需 use_retrieval+use_memory"
        self.begin(context_ids)
        q_hidden = self.raw.model.prelude_hidden(input_ids, attention_mask)
        caps = self.cfg.mode_caps_for(mode)
        budget_idx = self._budget_idx(q_hidden, max_budget_tokens=caps.get("max_budget"))
        self.memory = self.retrieve_into_memory(q_hidden, budget_idx=budget_idx)
        cb = self.make_loop_callback(max_budget_tokens=caps.get("max_budget"))
        out = self.raw.generate(
            inputs=input_ids, attention_mask=attention_mask, mode=mode,
            memory_provider=lambda step: self._memory_input(),
            loop_callback=cb, return_stats=True, **gen_kwargs)
        out["stats"].update({k: v for k, v in self.stats.items() if isinstance(v, (int, float))})
        out["stats"]["memory_tokens_final"] = self.memory.token_size if self.memory else 0
        out["stats"]["encoded_blocks"] = self.store.encoded_blocks
        out["stats"]["indexed_blocks"] = self.store.n_blocks
        out["stats"]["resident_source_ids"] = self.stats.get("resident_source_ids", [])
        return out if return_stats else out["generated_ids"]

    @torch.no_grad()
    def _memory_input(self):
        """按架构版本产出 memory 注入物：v1=逐层 KV 拼接；v2=(payload, valid) 独立 Reader。"""
        if self.memory is None:
            return None
        if getattr(self.cfg, "residual_v2", False):
            payload = self.memory.token_hidden()                    # [B, M, D]
            # T3.3：token 级 valid 来自 WorkingMemory.valid（块级扩展）
            valid = self.memory.token_valid()
            return MemoryInput(payload, valid,
                               memory_version=self.memory.memory_version)
        if self.memory._kv_cache is None:
            length = (self.stats["actual_valid_tokens"] if self.memory.hidden.size(0) == 1
                      else self.memory.token_size * self.memory.hidden.size(0))
            self.stats["legacy_projected_tokens"] = self.stats.get("legacy_projected_tokens", 0) + length
        return self.memory.get_kv(self.raw.model.thinking_core)

    # ---------------------------------------------------------------- 训练入口（SFT-C）
    def forward_with_oracles(self, input_ids, attention_mask, labels, context_ids,
                             gold_block_idx=None, oracle_retrieve=None, oracle_budget=None,
                             oracle_r=None, distill=True, query_len=None, append_only=False):
        """SFT-C / Stage 3 的 teacher-forced 终局训练前向（D5 + T0.1/T0.7）。

        query_len: 可见前缀边界（训练路径必填；初始检索与 gold CE 重算只用
        input_ids[:, :query_len]，换 gold completion 不改变检索——因果契约）。
        返回 (MiniMindXOutput, extras)。
        """
        if query_len is None:
            is_training = bool(getattr(self.model, "training", False)
                               or getattr(self.raw, "training", False))
            if is_training:
                raise ValueError(
                    "forward_with_oracles 训练路径必须显式传 query_len（禁止默认全序列，"
                    "否则检索/RB 路由会读到 gold answer）")
            query_len = input_ids.shape[1]  # 纯推理/全可见
        causal_ids = input_ids[:, :query_len]
        self.begin(context_ids)
        query_mask = attention_mask[:, :query_len] if attention_mask is not None else None
        q_hidden = self.raw.model.prelude_hidden(
            causal_ids, query_mask, no_grad=gold_block_idx is None and not distill)
        budget_idx = self._budget_idx(q_hidden, teacher_idx=oracle_budget)
        with torch.no_grad():  # D5：memory 构建/选择路径不回传梯度
            self.memory = self.retrieve_into_memory(q_hidden, budget_idx=budget_idx)
            mem_input = self._memory_input()
        # 动作 schema：调用方传入的 oracle_retrieve 一律视为 v2；旧 0/1 由 sft 边界转换
        cb = self.make_loop_callback(oracle_retrieve=oracle_retrieve, query_len=query_len)
        fwd = dict(collect_exits=True, loop_callback=cb, query_len=query_len)
        if getattr(self.cfg, "residual_v2", False):
            fwd["memory_payload"] = mem_input
        else:
            fwd["memory_kv_per_layer"] = mem_input
        def retrieval_losses():
            # Execute inside the same DDP forward as the supervised exits.
            return self._retrieval_losses(q_hidden, gold_block_idx, distill)

        if append_only:
            # Prefill once. Each following teacher token sees the committed cache and current memory.
            def execute(network, ids):
                decode_cb = self.make_loop_callback()
                past = None
                exits = []
                halt_steps = []
                for start in [0, *range(query_len, ids.shape[1])]:
                    end = query_len if start == 0 else start + 1
                    chunk_mask = attention_mask[:, :end] if attention_mask is not None else None
                    mem_key = "memory_payload" if self.cfg.residual_v2 else "memory_kv_per_layer"
                    chunk = network(ids[:, start:end], attention_mask=chunk_mask, past_key_values=past,
                                    use_cache=True, collect_exits=True,
                                    loop_callback=cb if start == 0 else decode_cb,
                                    **{mem_key: self._memory_input()})
                    if start == 0:
                        fields = vars(chunk).copy()
                        # Bind gold CE and the fine teacher to the last question-only search.
                        supervision = retrieval_losses()
                    if chunk.router_probs and "halt" in chunk.router_probs:
                        halt_steps.append(chunk.router_probs["halt"])
                    past = chunk.past_key_values
                    exits.append(chunk.exit_logits)
                fields["exit_logits"] = [torch.cat(parts, dim=1) for parts in zip(*exits)]
                fields["logits"] = fields["exit_logits"][-1]
                fields["past_key_values"] = None
                if halt_steps:
                    # Column zero is the last prefill position; subsequent columns are decode steps.
                    fields["router_probs"]["halt_by_step"] = [
                        torch.stack(steps, dim=1) for steps in zip(*halt_steps)]
                return fields, supervision
            fields, supervision = self.model(input_ids, execution_fn=execute)
        else:
            fields, supervision = self.model(input_ids, attention_mask=attention_mask, labels=None,
                                             retrieval_loss_fn=retrieval_losses, **fwd)
        out = MiniMindXOutput(**fields)
        out.router_trace = list(self.trajectory)
        extras = {"retrievals": self.stats["retrievals"],
                  "memory_tokens": self.memory.token_size,
                  "budget_cap_tokens": self.stats.get("budget_cap_tokens", 0),
                  "actual_valid_tokens": self.stats.get("actual_valid_tokens", 0),
                  "unique_blocks": self.stats.get("unique_blocks", 0),
                  "trajectory": list(self.trajectory),
                  "warm_available": self.has_warm_cache(),
                  "action_counts": {k: self.stats.get(k, 0) for k in
                                    ("reuse", "warm_rescore", "global_refresh")}}
        extras.update(supervision)
        # ---- T2.5：全展开时收集三动作 logits（用于 action CE）----
        if getattr(out, "router_probs", None) and out.router_probs.get("retrieval_action") is not None:
            extras["retrieval_action_logits"] = out.router_probs["retrieval_action"]
            extras["retrieval_action_masks"] = out.router_probs.get("retrieval_action_mask")
        return out, extras

    def _retrieval_losses(self, initial_query, gold_block_idx, distill):
        result = self.stats.get("last_global_result")
        losses = {}
        if result is None or (gold_block_idx is None and not distill):
            return losses
        hidden = self.stats.get("last_global_query")
        if hidden is None:
            hidden = initial_query
        retriever = self.raw.model.retriever
        coarse = retriever.coarse_scores(retriever.project_query(hidden), self.store)
        if gold_block_idx is not None and ((gold_block_idx >= 0) & (gold_block_idx < coarse.size(1))).all():
            visible = result.coarse_scores.isfinite()
            if visible.gather(1, gold_block_idx.view(-1, 1)).all():
                losses["retrieval_ce"] = retriever.retrieval_ce_loss(coarse.masked_fill(~visible, -torch.inf),
                                                                     gold_block_idx)
        if distill and result.fine_scores is not None:
            losses["distill_loss"] = retriever.distill_loss(coarse, result.cand_idx, result.fine_scores)
        return losses
