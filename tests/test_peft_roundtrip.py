"""Phase 0 gate test: PEFT save/reload parity + disable_adapter() toggle.

Run on GPU:
    UNSLOTH_CE_LOSS_TARGET_GB=4 python tests/test_peft_roundtrip.py
"""

import os
os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "4"
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")

import shutil
import sys
import torch


def main():
    from unsloth import FastModel
    from peft import PeftModel

    SAVE_DIR = "/tmp/peft_roundtrip_test"
    MODEL = "Qwen/Qwen3.5-0.8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/5] Loading {MODEL} via FastModel...")
    model, tokenizer = FastModel.from_pretrained(
        MODEL, max_seq_length=2048, load_in_4bit=False, dtype=torch.bfloat16,
    )
    model = FastModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=False,
    )

    # Perturb LoRA-B so adapter is non-trivial
    print("[2/5] Perturbing LoRA-B weights...")
    n_perturbed = 0
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            param.data.normal_(0, 0.01)
            n_perturbed += 1
    print(f"  Perturbed {n_perturbed} LoRA-B parameters")

    # Qwen3.5 tokenizer from FastModel is a Qwen3VLProcessor, not a
    # standard tokenizer. Access the underlying tokenizer for encode().
    inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
    input_text = "What is the capital of France?"
    input_ids = inner_tok.encode(input_text, return_tensors="pt").to(device)
    model.eval()

    with torch.no_grad():
        logits_before = model(input_ids=input_ids).logits.float().cpu()

    # Save
    print(f"[3/5] Saving to {SAVE_DIR}...")
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    # Reload via PEFT
    print("[4/5] Reloading via PeftModel.from_pretrained...")
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=device,
    )
    reloaded = PeftModel.from_pretrained(base, SAVE_DIR, is_trainable=True)
    reloaded.eval()

    with torch.no_grad():
        logits_after = reloaded(input_ids=input_ids).logits.float().cpu()

    # Check parity
    max_diff = (logits_before - logits_after).abs().max().item()
    print(f"  Max logit difference: {max_diff:.6f}")
    assert max_diff < 1e-2, f"PEFT round-trip FAILED: max diff = {max_diff}"
    print("  PASS: PEFT round-trip parity")

    # Test disable_adapter()
    print("[5/5] Testing disable_adapter()...")
    with torch.no_grad():
        logits_with = reloaded(input_ids=input_ids).logits.float().cpu()
        with reloaded.disable_adapter():
            logits_without = reloaded(input_ids=input_ids).logits.float().cpu()

    adapter_diff = (logits_with - logits_without).abs().max().item()
    print(f"  Adapter effect (max diff): {adapter_diff:.6f}")
    assert adapter_diff > 1e-3, (
        f"disable_adapter() FAILED: max diff = {adapter_diff} — "
        f"adapter has no effect (LoRA-B was perturbed, so this should differ)"
    )
    print("  PASS: disable_adapter() toggles output")

    # Cleanup
    shutil.rmtree(SAVE_DIR)
    print("\nAll gate tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
