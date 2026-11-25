"""Sample or replay one append-only MiniMind-X trajectory, including controller decisions."""
import torch
from model.minimind_x.controller import AdaptiveController


def trajectory(model, prompt, context, max_tokens, temperature, eos_id, fixed_depth=None,
               replay=None, visibility=None, verify_behavior=False):
    if temperature <= 0 or max_tokens < 1:
        raise ValueError("temperature and generation length must be positive")
    cfg = model.config
    ctrl = None
    if context is not None and context.numel() and cfg.use_retrieval:
        ctrl = AdaptiveController(model, block_visible_from=visibility)
        ctrl.begin(context)
    past, chunk = None, prompt
    records, joint_logps = [], []
    memory_reads = 0
    for position in range(len(replay) if replay is not None else max_tokens):
        expected = replay[position]["events"] if replay is not None else None
        events, logps, memory_states = [], [], []

        def choose(kind, logits):
            # Full-support categorical for both behavior and reference, including the temperature.
            dist = torch.log_softmax(logits.float().reshape(-1), dim=0)
            if expected is None:
                value = int(torch.multinomial(dist.detach().exp(), 1))
            else:
                if len(events) >= len(expected) or expected[len(events)][0] != kind:
                    raise ValueError(f"Trajectory event mismatch at token {position}: {kind}")
                value = expected[len(events)][1]
            selected = dist[value]
            if not torch.isfinite(selected):
                raise ValueError(f"Unavailable {kind} action {value}")
            if verify_behavior:
                torch.testing.assert_close(selected.detach(), selected.new_tensor(expected[len(events)][2]),
                                           rtol=1e-5, atol=1e-5)
            events.append((kind, value, float(selected.detach())))
            logps.append(selected)
            return value

        def budget(hidden):
            if fixed_depth is not None:
                return ctrl._budget_idx(hidden)
            return choose("budget", model.model.budget_router(hidden))

        if ctrl is not None and position == 0:
            hidden = model.model.prelude_hidden(prompt)
            idx = budget(hidden)
            with torch.no_grad():
                ctrl.memory = ctrl.retrieve_into_memory(hidden, idx)

        cap = fixed_depth or cfg.max_loop
        if not cfg.use_loop:
            cap = 1
        halt_round = 0

        def halt(prob):
            nonlocal halt_round
            halt_round += 1
            if halt_round == cap:
                return torch.ones_like(prob)
            prob = prob.float().clamp(1e-7, 1 - 1e-7)
            action = choose("halt", torch.stack((torch.log1p(-prob), prob.log())))
            return torch.full_like(prob, action)

        def retrieve(r, hidden):
            nonlocal memory_reads
            ctrl.store.advance_round(r + 1)
            action = choose("retrieval", model.model.retrieve_router.action_logits(hidden, ctrl.has_warm_cache()))
            if action:
                idx = budget(hidden)
                with torch.no_grad():
                    ctrl.memory = ctrl.execute_retrieval_action(action, hidden, idx, ctrl.memory)
                ctrl.stats["re_retrievals"] += 1
            memory_reads += ctrl.stats["actual_valid_tokens"] * chunk.size(1)
            memory_states.append([ctrl.memory.memory_version, ctrl.stats["resident_source_ids"]])
            return ctrl._memory_input()

        mem_kw = {}
        if ctrl is not None:
            memory_states.append([ctrl.memory.memory_version, ctrl.stats["resident_source_ids"]])
            memory_reads += ctrl.stats["actual_valid_tokens"] * chunk.size(1)
            key = "memory_payload" if cfg.residual_v2 else "memory_kv_per_layer"
            mem_kw[key] = ctrl._memory_input()
        # Match deployment's fixed-cap prompt prefill. All stochastic choices are recorded.
        adaptive = fixed_depth is None and cfg.use_loop and (
            position > 0 or getattr(cfg, "prefill_policy", "fixed_cap") != "fixed_cap")
        out = model(chunk, past_key_values=past, use_cache=True, logits_to_keep=1,
                    num_loops=cap, halt=adaptive, halt_decision=halt,
                    loop_callback=retrieve if ctrl is not None else None, **mem_kw)
        token = choose("token", out.logits[0, -1] / temperature)
        depth = out.num_loops_executed
        if expected is not None and (len(events) != len(expected) or depth != replay[position]["depth"]):
            raise ValueError(f"Trajectory depth/event count mismatch at token {position}")
        if expected is not None and token != replay[position]["token"]:
            raise ValueError(f"Trajectory token mismatch at token {position}")
        if verify_behavior and memory_states != replay[position]["memory_states"]:
            raise ValueError(f"Evidence changed during behavior replay at token {position}")
        records.append(dict(events=events, token=token, depth=depth, memory_states=memory_states))
        joint_logps.append(torch.stack(logps).sum())
        past, chunk = out.past_key_values, prompt.new_tensor([[token]])
        if replay is None and token == eos_id:
            break
    stats = ctrl.stats if ctrl else {}
    return records, torch.stack(joint_logps), memory_reads, stats


def replay_logps(model, prompt, context, records, temperature, fixed_depth=None, visibility=None, verify_behavior=False):
    def execute(network, ids):
        return trajectory(network, ids, context, len(records), temperature, None,
                          fixed_depth, replay=records, visibility=visibility, verify_behavior=verify_behavior)[1]
    return model(prompt, execution_fn=execute)
