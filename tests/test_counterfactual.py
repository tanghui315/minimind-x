import copy
import itertools

import pytest
import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from scripts.eval_counterfactual import (ACTIONS, branch, capture, evidence, fork_controller,
                                         matched_em, matched_logp, memory_kw, prepare, score)


def request(v2=True, empty=False, rounds=3):
    torch.manual_seed(197)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=96, max_loop=rounds,
                          block_size=8, memory_budgets=[8,16,24], default_budget_idx=0,
                          variant="minimind_x", residual_v2=v2, topk_coarse=3, flash_attn=False)
    model = MiniMindXForCausalLM(cfg).eval()
    q, delimiter = torch.tensor([[1,5,6,7]]), torch.tensor([[8,9]])
    ctx = torch.arange(12,74) if not empty else torch.empty(0,dtype=torch.long)
    ctrl, past = prepare(model, q, delimiter, ctx, None, budget_idx=0)
    return ctrl, past, delimiter[:, -1:]


@pytest.mark.parametrize("v2",[True,False])
@pytest.mark.parametrize("r",[1,2])
@torch.no_grad()
def test_fork_matches_full_replay_and_routed_choice(v2,r):
    ctrl,past,x = request(v2)
    original = fork_controller(ctrl)
    dec = capture(ctrl,x,past,r)
    anchor = dec.prefix.core["exit_hiddens"][-1].clone()
    for action in reversed(ACTIONS):
        fork = fork_controller(dec.ctrl)
        actual = branch(dec,fork,action)
        reference = fork_controller(original)
        default_cb = reference.make_loop_callback()
        override = reference.make_loop_callback([0]*(r-1)+[max(0,ACTIONS.index(action)-1)])
        def cb(i,h):
            return (override if i==r-1 else default_cb)(i,h)
        halt_calls = 0
        def halt(p):
            nonlocal halt_calls
            halt_calls += 1
            return torch.zeros_like(p) if halt_calls <= r else p
        expected = original.raw(x,past_key_values=past,use_cache=True,logits_to_keep=1,
            num_loops=r if action=="stop" else ctrl.cfg.max_loop,
            halt=action!="stop",halt_decision=halt,loop_callback=cb,**memory_kw(reference))
        torch.testing.assert_close(actual.logits,expected.logits)
        torch.testing.assert_close(actual.past_key_values,expected.past_key_values)
        assert actual.num_loops_executed == expected.num_loops_executed
        assert evidence(fork) == evidence(reference)
        torch.testing.assert_close(torch.tensor(score(fork,actual,torch.tensor([14,15]))),
                                   torch.tensor(score(reference,expected,torch.tensor([14,15]))))
    torch.testing.assert_close(dec.prefix.core["exit_hiddens"][-1],anchor)
    assert len(dec.prefix.core["presents_per_pass"]) == r
    if r == 1:
        router = branch(dec,fork_controller(ctrl),dec.router)
        native = original.raw(x,past_key_values=past,use_cache=True,halt=True,num_loops=3,
                              logits_to_keep=1,loop_callback=original.make_loop_callback(),**memory_kw(original))
        torch.testing.assert_close(router.logits,native.logits)


@torch.no_grad()
def test_mask_and_branch_cache_isolation():
    ctrl,past,x = request(empty=True)
    dec = capture(ctrl,x,past,1)
    assert "rescore_warm" not in dec.legal
    with pytest.raises(ValueError,match="Unavailable"):
        branch(dec,fork_controller(ctrl),"rescore_warm")
    terminal = capture(fork_controller(ctrl),x,past,3)
    assert terminal.legal == ("stop",)
    with pytest.raises(ValueError,match="Unavailable"):
        branch(terminal,ctrl,"reuse")
    ctrl,past,x=request()
    dec=capture(ctrl,x,past,1)
    original_events=copy.deepcopy(ctrl.trajectory)
    original_keys=set(ctrl.store._hidden_cache)
    assert original_keys
    cold=fork_controller(ctrl,cold=True)
    assert not cold.store._hidden_cache and cold.has_warm_cache()
    hot=fork_controller(ctrl)
    hot_out=branch(dec,hot,"global_refresh")
    cold_out=branch(dec,cold,"global_refresh")
    torch.testing.assert_close(hot_out.logits,cold_out.logits)
    assert cold.store.encoded_blocks>ctrl.store.encoded_blocks
    assert set(ctrl.store._hidden_cache)==original_keys and ctrl.trajectory==original_events
    assert hot.store is not ctrl.store and cold.store._hidden_cache is not hot.store._hidden_cache


@torch.no_grad()
def test_stop_does_not_rerun_core_or_prelude(monkeypatch):
    ctrl,past,x=request(rounds=2)
    d=capture(ctrl,x,past,1)
    calls=[]
    core=ctrl.raw.model.thinking_core
    run=core._single_pass
    def count(*a,**kw):
        calls.append(1)
        return run(*a,**kw)
    def repeated(*args,**kwargs):
        raise AssertionError("repeated shared prelude")
    monkeypatch.setattr(core,"_single_pass",count)
    monkeypatch.setattr(ctrl.raw.model.prelude,"forward",repeated)
    branch(d,fork_controller(ctrl),"stop")
    assert not calls
    branch(d,fork_controller(ctrl),"reuse")
    assert len(calls)==1


def test_budgeted_upper_bounds_match_enumeration():
    def arm(ms,hit,logp):
        return dict(generation_ms=ms,correct=hit,mean_logp=logp)
    rows=[dict(actions=dict(stop=arm(1,0,-2),reuse=arm(4,1,-1))),
          dict(actions=dict(stop=arm(1,0,-3),reuse=arm(2,1,-1.5))),
          dict(actions=dict(stop=arm(3,1,-4),reuse=arm(1,0,-1)))]
    for budget in (3,4,5,6,7,8,9):
        possible=[c for c in itertools.product(*(r["actions"].values() for r in rows))
                  if sum(x["generation_ms"] for x in c)<=budget]
        assert matched_em(rows,budget)["correct"]==max(sum(x["correct"] for x in c) for c in possible)
        assert matched_logp(rows,budget)["mean_logp"]==pytest.approx(
            max(sum(x["mean_logp"] for x in c)/3 for c in possible))


def test_cli_scores_only_validation_and_keeps_gold_out_of_state(tmp_path,monkeypatch):
    import json
    import sys
    from scripts import eval_counterfactual as cf
    from trainer.x_utils import save_x_checkpoint
    tok=cf.AutoTokenizer.from_pretrained(cf.ROOT/"model")
    cfg=MiniMindXConfig(hidden_size=32,num_hidden_layers=6,vocab_size=len(tok),
                        variant="minimind_x",residual_v2=True,max_loop=2,block_size=8,
                        memory_budgets=[8,16,24],max_position_embeddings=128)
    checkpoint=save_x_checkpoint(MiniMindXForCausalLM(cfg),tmp_path,"cf",32)
    sample=dict(split="val",question="编号？",context="旧记录。"*4+"编号37。",family="qa")
    data=tmp_path/"val.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in [
        {"split":"train"},dict(sample,answer="37"),dict(sample,answer="91")]))
    output=tmp_path/"report.json"
    monkeypatch.setattr(sys,"argv",["eval","--checkpoint",checkpoint,"--data",str(data),"--out",str(output),
                                    "--device","cpu","--repeats","1","--max-new-tokens","2","--cache","both"])
    cf.main()
    report=json.loads(output.read_text())
    assert all(report["summary"][mode]["n"]==2 for mode in ["cold","request"])
    assert set(report["full_policies"])=={"fixed_R1","fixed_R2","initial_only","router"}
    a,b=report["samples"][::2]
    assert a["router"]==b["router"] and a["initial_sources"]==b["initial_sources"]
    for action in ACTIONS:
        arm,other=a["actions"][action],b["actions"][action]
        assert arm["generated_ids"]==other["generated_ids"]
        assert arm["source_ids_at_exit"]==other["source_ids_at_exit"]
        assert arm["mean_logp"]!=other["mean_logp"]
        assert len(arm["decision_ms_samples"])==1 and arm["generation_ms"]>0
