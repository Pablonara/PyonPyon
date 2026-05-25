"""Train Qwen3.5 LoRA SFT on JSONL traces."""

from __future__ import annotations

import argparse
import os

os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "4"
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))

from qwen3_rl.sft.jsonl import jsonl_to_text_records, make_dataset, token_length_stats


LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _str_to_grad_ckpt(value: str):
    lowered = value.lower()
    if lowered in {"false", "0", "off", "none"}:
        return False
    if lowered in {"true", "1", "on"}:
        return True
    if lowered == "unsloth":
        return "unsloth"
    raise argparse.ArgumentTypeError("expected false, true, or unsloth")


def _load_template(path: str | None) -> str | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _unwrap_tokenizer(tokenizer):
    return tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./neko-data.jsonl",
                        help="JSONL input with messages, text, or rollout trace raw_text")
    parser.add_argument("--schema", choices=["auto", "messages", "text", "rollout_trace"],
                        default="auto")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--template", default="./qwen35_chat_template.jinja")
    parser.add_argument("--output-dir", default="./sft_output")
    parser.add_argument("--prepared-output-dir", default=None,
                        help="Optional save_to_disk path for prepared text dataset")
    parser.add_argument("--prepared-dataset", default=None,
                        help="Load a previously save_to_disk prepared text dataset")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--min-reward", type=float, default=None,
                        help="For rollout traces, keep rows with reward >= value")

    parser.add_argument("--max-seq-len", type=int, default=77695)
    parser.add_argument("--auto-max-seq-len", action="store_true",
                        help="Set max_seq_len to max observed token length + 1")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--grad-checkpointing", type=_str_to_grad_ckpt, default=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-packing", action="store_true")
    parser.add_argument("--no-response-mask", action="store_true",
                        help="Train on every token instead of assistant responses only")
    parser.add_argument("--report-to", default="none",
                        help="Trainer reporting target, e.g. none or wandb")
    parser.add_argument("--run-name", default=None,
                        help="Optional run name for W&B / Trainer logs")
    parser.add_argument("--wandb-project", default=None,
                        help="Set WANDB_PROJECT for this run")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Upload the final adapter to the Hugging Face Hub")
    parser.add_argument("--hub-model-id", default=None,
                        help="Hugging Face repo id for --push-to-hub")
    parser.add_argument("--hub-private", action="store_true",
                        help="Create/upload to a private Hugging Face repo")
    parser.add_argument("--no-compile-warmup", action="store_true")
    return parser


def _load_prepare_tokenizer(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=os.environ["HF_HOME"])
    template = _load_template(args.template)
    if template is not None:
        tokenizer.chat_template = template
    return tokenizer


def _prepare_dataset(args, tokenizer):
    if args.prepared_dataset:
        from datasets import load_from_disk

        dataset = load_from_disk(args.prepared_dataset)
        stats = token_length_stats(dataset["text"], tokenizer)
        print(
            "[sft:data] "
            f"loaded={args.prepared_dataset} rows={stats['count']} "
            f"tok_min={stats['min']} tok_median={stats['median']} "
            f"tok_max={stats['max']}"
        )
        return dataset, stats

    records = jsonl_to_text_records(
        args.data,
        tokenizer=tokenizer,
        schema=args.schema,
        max_rows=args.max_examples,
        min_reward=args.min_reward,
    )
    dataset = make_dataset(records)
    stats = token_length_stats([record["text"] for record in records], tokenizer)
    print(
        "[sft:data] "
        f"rows={stats['count']} tok_min={stats['min']} "
        f"tok_median={stats['median']} tok_max={stats['max']}"
    )
    if args.prepared_output_dir:
        dataset.save_to_disk(args.prepared_output_dir)
        print(f"[sft:data] saved prepared dataset to {args.prepared_output_dir}")
    return dataset, stats


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    if args.prepare_only:
        tokenizer = _load_prepare_tokenizer(args)
        _prepare_dataset(args, tokenizer)
        return

    dataset = None
    stats = None
    if args.auto_max_seq_len:
        prepare_tokenizer = _load_prepare_tokenizer(args)
        dataset, stats = _prepare_dataset(args, prepare_tokenizer)
        args.max_seq_len = int(stats["max"]) + 1
        print(f"[sft] auto max_seq_len={args.max_seq_len}")

    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer
    import torch

    print(f"[sft] Loading {args.model}...")
    model, tokenizer = FastModel.from_pretrained(
        args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.load_in_4bit,
        cache_dir=os.environ["HF_HOME"],
    )
    tokenizer = _unwrap_tokenizer(tokenizer)
    template = _load_template(args.template)
    if template is not None:
        tokenizer.chat_template = template
        print(f"[sft] Loaded chat template from {args.template}")

    if dataset is None or stats is None:
        dataset, stats = _prepare_dataset(args, tokenizer)
    max_seq_len = args.max_seq_len

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=LORA_TARGETS,
        lora_alpha=args.lora_alpha or args.lora_r,
        lora_dropout=0,
        use_gradient_checkpointing=args.grad_checkpointing,
    )

    if not args.no_compile_warmup and torch.cuda.is_available():
        print("[sft] Compiling kernels with dummy forward+backward...")
        device = next(model.parameters()).device
        vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
        dummy_ids = torch.randint(0, vocab_size, (1, min(128, max_seq_len)), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=dummy_ids, labels=dummy_ids.clone())
            out.loss.backward()
        model.zero_grad(set_to_none=True)
        del dummy_ids, out
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        packing=not args.no_packing,
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    if not args.no_response_mask:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    print(
        f"[sft] Starting train: rows={len(dataset)} max_seq_len={max_seq_len} "
        f"bs={args.batch_size} accum={args.grad_accum} epochs={args.epochs}"
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[sft] saved to {args.output_dir}")
    if args.push_to_hub:
        trainer.push_to_hub()
        print(f"[sft] pushed to hub: {args.hub_model_id or args.output_dir}")


if __name__ == "__main__":
    main()
