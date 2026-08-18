"""Convert a transformers WhisperForConditionalGeneration checkpoint (as
saved by finetune_whisper.py into whisper_finetuned/) into the openai-whisper
.pt format that whisper.load_model(path) expects.

Usage:
    python convert_finetuned_to_openai.py
Output: ../whisper_finetuned/model.pt
"""
import os
import re

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(HERE, "..", "whisper_finetuned"))
OUT = os.path.join(SRC_DIR, "model.pt")

# HF key -> openai-whisper key. "{i}" is the layer index, "{p}" the projection.
_MAP = [
    (re.compile(r"^model\.encoder\.conv1\.(.+)$"), r"encoder.conv1.\1"),
    (re.compile(r"^model\.encoder\.conv2\.(.+)$"), r"encoder.conv2.\1"),
    (re.compile(r"^model\.encoder\.embed_positions\.weight$"), r"encoder.positional_embedding"),
    (re.compile(r"^model\.encoder\.layer_norm\.(.+)$"), r"encoder.ln_post.\1"),
    (re.compile(r"^model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.(.+)$"), r"encoder.blocks.\1.attn.query.\2"),
    (re.compile(r"^model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.(.+)$"), r"encoder.blocks.\1.attn.key.\2"),
    (re.compile(r"^model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.(.+)$"), r"encoder.blocks.\1.attn.value.\2"),
    (re.compile(r"^model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.(.+)$"), r"encoder.blocks.\1.attn.out.\2"),
    (re.compile(r"^model\.encoder\.layers\.(\d+)\.self_attn_layer_norm\.(.+)$"), r"encoder.blocks.\1.attn_ln.\2"),
    (re.compile(r"^model\.encoder\.layers\.(\d+)\.fc1\.(.+)$"), r"encoder.blocks.\1.mlp.0.\2"),
    (re.compile(r"^model\.encoder\.layers\.(\d+)\.fc2\.(.+)$"), r"encoder.blocks.\1.mlp.2.\2"),
    (re.compile(r"^model\.encoder\.layers\.(\d+)\.final_layer_norm\.(.+)$"), r"encoder.blocks.\1.mlp_ln.\2"),
    (re.compile(r"^model\.decoder\.embed_tokens\.weight$"), r"decoder.token_embedding.weight"),
    (re.compile(r"^model\.decoder\.embed_positions\.weight$"), r"decoder.positional_embedding"),
    (re.compile(r"^model\.decoder\.layer_norm\.(.+)$"), r"decoder.ln.\1"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn\.q_proj\.(.+)$"), r"decoder.blocks.\1.attn.query.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn\.k_proj\.(.+)$"), r"decoder.blocks.\1.attn.key.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn\.v_proj\.(.+)$"), r"decoder.blocks.\1.attn.value.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.(.+)$"), r"decoder.blocks.\1.attn.out.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.q_proj\.(.+)$"), r"decoder.blocks.\1.cross_attn.query.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.k_proj\.(.+)$"), r"decoder.blocks.\1.cross_attn.key.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.v_proj\.(.+)$"), r"decoder.blocks.\1.cross_attn.value.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.out_proj\.(.+)$"), r"decoder.blocks.\1.cross_attn.out.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn_layer_norm\.(.+)$"), r"decoder.blocks.\1.attn_ln.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.(.+)$"), r"decoder.blocks.\1.cross_attn_ln.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.fc1\.(.+)$"), r"decoder.blocks.\1.mlp.0.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.fc2\.(.+)$"), r"decoder.blocks.\1.mlp.2.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.final_layer_norm\.(.+)$"), r"decoder.blocks.\1.mlp_ln.\2"),
]

DIMS = {
    "n_mels": 80,
    "n_audio_ctx": 1500,
    "n_audio_state": 768,
    "n_audio_head": 12,
    "n_audio_layer": 12,
    "n_vocab": 51865,
    "n_text_ctx": 448,
    "n_text_state": 768,
    "n_text_head": 12,
    "n_text_layer": 12,
}


def to_openai_key(hf_key: str) -> str:
    for pattern, repl in _MAP:
        m = pattern.match(hf_key)
        if m:
            return re.sub(pattern, repl, hf_key)
    raise KeyError(f"no mapping for '{hf_key}'")


def main():
    safetensors_path = os.path.join(SRC_DIR, "model.safetensors")
    if not os.path.isfile(safetensors_path):
        raise FileNotFoundError(f"{safetensors_path} missing")

    state = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        embed_tokens = None
        for hf_key in f.keys():
            tensor = f.get_tensor(hf_key)
            openai_key = to_openai_key(hf_key)
            state[openai_key] = tensor
            if hf_key == "model.decoder.embed_tokens.weight":
                embed_tokens = tensor

    # decoder.proj_out.weight is tied to token_embedding in openai-whisper, so
    # it must be omitted; loading token_embedding fills the shared parameter.

    torch.save({"dims": DIMS, "model_state_dict": state}, OUT)
    print(f"[convert] wrote {OUT} ({len(state)} tensors)")


if __name__ == "__main__":
    main()
