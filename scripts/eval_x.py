"""MiniMind-X 推理入口（任务清单 T7.5；三模式 Fast/Normal/Deep + 终局长上下文）。"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, TextStreamer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from trainer.x_utils import x_weight_path, tokenize_tensors
from trainer.trainer_utils import setup_seed
import random

PROMPTS = ["你有什么特长？", "为什么天空是蓝色的", "解释什么是机器学习", "1+2*3等于几？"]


def main():
    p = argparse.ArgumentParser(description="MiniMind-X 推理与对话")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="sft_x_c")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="minimind_x")
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--mode", type=str, default=None, choices=["fast", "normal", "deep"],
                   help="推理模式（None=自适应 halting）")
    p.add_argument("--fixed_loops", type=int, default=None, help="固定 R（覆盖模式）")
    p.add_argument("--context_file", type=str, default=None,
                   help="长上下文文件 → 终局检索路径（D9）")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--historys", type=int, default=0)
    p.add_argument("--show_speed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = MiniMindXConfig(preset=args.preset, variant=args.variant,
                          **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                   num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    model = MiniMindXForCausalLM(cfg)
    model.load_state_dict(torch.load(x_weight_path(args.weight_dir, args.weight, cfg.hidden_size),
                                     map_location="cpu"), strict=False)
    model = model.half().eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    ctrl = AdaptiveController(model) if args.context_file else None
    context_ids = None
    if args.context_file:
        text = open(args.context_file, encoding="utf-8").read()
        context_ids = torch.tensor(
            tokenizer(text, add_special_tokens=False).input_ids, device=args.device)

    conversation = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    for prompt in PROMPTS:
        setup_seed(random.randint(0, 314159))
        print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        inputs_txt = tokenizer.apply_chat_template(conversation, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = tokenize_tensors(tokenizer, inputs_txt, args.device, truncation=True)
        print('🧠: ', end='')
        import time
        st = time.time()
        gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                      streamer=streamer, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id, top_p=args.top_p,
                      temperature=args.temperature, repetition_penalty=1,
                      mode=args.mode, fixed_loops=args.fixed_loops, return_stats=True)
        if ctrl is not None:
            out = ctrl.generate_with_context(inputs["input_ids"], inputs["attention_mask"],
                                             context_ids=context_ids, **gen_kw)
            gen_ids = out["generated_ids"]
        else:
            out = model.generate(inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                 **gen_kw)
            gen_ids = out["generated_ids"] if isinstance(out, dict) else out
        response = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:].tolist(),
                                    skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        stats = out.get("stats", {}) if isinstance(out, dict) else {}
        n_tok = gen_ids.shape[1] - inputs["input_ids"].shape[1]
        if args.show_speed:
            print(f'\n[Speed]: {n_tok / max(time.time() - st, 1e-6):.2f} tokens/s | '
                  f'avg_loops: {stats.get("avg_loops", "-")}')


if __name__ == "__main__":
    main()
