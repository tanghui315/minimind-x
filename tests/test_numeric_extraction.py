import random

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from scripts import eval_fact_swap as fs
from trainer import train_numeric_extraction as numeric


def test_extraction_prompts_match_evaluation_and_keep_questions_paired():
    tok = AutoTokenizer.from_pretrained(fs.ROOT / "model")
    sites, values = [301], ["5237", "7926", "4119"]
    pairs = numeric.generate_examples(random.Random(7), tok, 3, sites, values)
    for pair in pairs:
        questions = []
        answers = []
        for prefix, delimiter, target in pair:
            prompt = tok.decode(prefix[1:])
            facts, question = prompt.split("\n问题：")
            answer = tok.decode(target[:-1])
            assert prefix[0] == tok.bos_token_id and target[-1] == tok.eos_token_id
            assert tok.decode(delimiter) == "答案："
            assert facts.startswith("资料：") and answer in facts and answer not in question
            assert "-301" in question and answer in values
            questions.append(question)
            answers.append(answer)
        assert questions[0] == questions[1] and answers[0] != answers[1]


def test_numeric_loss_matches_unpadded_answer_only_causal_ce_and_skips_reader():
    torch.manual_seed(11)
    config = MiniMindXConfig(
        variant="minimind_x",
        residual_v2=True,
        hidden_size=32,
        vocab_size=80,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        max_loop=2,
        dropout=0.0,
    )
    model = MiniMindXForCausalLM(config).eval()
    batch = [([1, 4, 5, 6], [10, 11], [12, 13, 2]), ([1, 7], [10, 11], [19, 2])]
    expected = 0.0
    for prefix, delimiter, target in batch:
        seq = prefix + delimiter + target[:-1]
        logits = model(torch.tensor([seq]), num_loops=2).logits[0, len(prefix + delimiter) - 1 :]
        expected += F.cross_entropy(logits, torch.tensor(target), reduction="sum").item()
    loss = numeric.batch_loss(model, batch, "cpu", loops=2)
    torch.testing.assert_close(loss.detach(), torch.tensor(expected / 5))
    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert all(p.grad is None for p in model.model.thinking_core.memory_reader.parameters())
    assert all(p.grad is None for p in model.model.retriever.parameters())
