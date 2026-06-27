"""Convert bf16 unlimited-ocr to hybrid: vision encoder in bf16, LLM decoder in 8-bit."""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import quantize_model

from mlx_vlm.utils import fetch_from_hub, save_config, save_weights


def main():
    parser = argparse.ArgumentParser(
        description="Convert unlimited-ocr to hybrid (bf16 vision + q8 LLM)"
    )
    parser.add_argument("--hf-path", required=True, help="Path to bf16 MLX model")
    parser.add_argument("--mlx-path", default="mlx_model_hybrid", help="Output path")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Bits for LLM quantization")
    args = parser.parse_args()

    src = Path(args.hf_path)
    dst = Path(args.mlx_path)

    print("[INFO] Loading bf16 model...")
    model, config, processor = fetch_from_hub(src, lazy=False)

    def quant_predicate(path: str, _module) -> bool:
        # Skip vision/SAM/projector — keep them in bf16
        skip_prefixes = ("vision_model", "sam_model", "projector")
        if any(path.startswith(p) for p in skip_prefixes):
            return False
        # image_newline and view_separator are bare mx.array, not nn.Module,
        # so they won't have to_quantized() — quantize_model skips them.
        return True

    print(f"[INFO] Quantizing LLM decoder to {args.bits}-bit...")
    qconfig = {"group_size": 64, "bits": args.bits}

    model, config = quantize_model(
        model,
        config,
        group_size=qconfig["group_size"],
        bits=qconfig["bits"],
        quant_predicate=quant_predicate,
    )

    # Also ensure vision modules are skipped during future quantized loading
    if "quantization" in config:
        config["quantization"]["skip_vision"] = True

    print(f"[INFO] Saving to {dst}...")
    save_weights(dst, model, donate_weights=True)
    save_config(config, config_path=dst / "config.json")

    for pattern in ["*.py", "*.json"]:
        for f in src.glob(pattern):
            if f.name in ("model.safetensors.index.json", "config.json"):
                continue
            shutil.copy(f, dst)

    for item in src.iterdir():
        if item.is_dir():
            dest = dst / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    processor.save_pretrained(dst)

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
