# Unlimited-OCR MLX Port

The `unlimited_ocr` model is ported to MLX and working.

## Models

- **bf16**: `/Volumes/Realtek/models/mlx/unlimited-ocr-bf16-mlx` — **best**, output matches PT reference, 10.3GB peak
- **hybrid-8bit**: `/Volumes/Realtek/models/mlx/unlimited-ocr-hybrid-mlx` — bf16 vision + q8 LLM, matches bf16 quality, 7.5GB peak
- **hybrid-4bit**: `/Volumes/Realtek/models/mlx/unlimited-ocr-hybrid-4bit-mlx` — bf16 vision + q4 LLM, minor text degradation, 6.1GB peak
- **8-bit** (full): `/Volumes/Realtek/models/mlx/unlimited-ocr-8bit-mlx` — works but degenerates late
- **4-bit** (full): `/Volumes/Realtek/models/mlx/unlimited-ocr-4bit-mlx` — too aggressive quantization, unusable

## Usage

```bash
python run_unlimited_ocr.py
```

## Fixes Applied

- **Token loading**: `DeepseekOCRProcessor.from_pretrained` in `processing_deepseekocr.py` patched to use raw `Tokenizer.from_file()` instead of `LlamaTokenizerFast.from_pretrained` (corrupts BPE on transformers >= 5.x)
- **Image file**: Use `paper_test.jpg` (RGB) instead of `paper_test.png` (RGBA) to avoid pixel diff
- **Processor registration**: `UnlimitedOCRProcessor` registered for `deepseekocr` model type in `processing_unlimitedocr.py`

## R-SWA (Ring Sliding Window Attention)

Implemented in `language.py` via `RingSlidingKVCache`. Matches upstream behavior:
- **Prefill**: full causal attention (all prompt tokens stored)
- **Warmup decode**: normal KVCache append, full attention over all cached tokens
- **Steady-state decode**: ring buffer overwrites old slots, attention limited to `prefill_length + window_size` tokens, mask=None (correct for q_len=1)

## Known Issues (minor)

- Stray `ovi` prefix appears with quantized models (not with bf16)
- Pixel values patches have small differences between macOS and Linux PIL (resize interpolation)

## Building the bf16 model

```bash
python -m mlx_vlm convert --hf-path <path-to-original-model> --mlx-path <output> --dtype bfloat16
```

## PyTorch Reference (on Linux server)

Access: `ssh hp` → working dir `/home/jason/prj/unlimited-ocr`
Env: `conda activate /media/zt/prj/env/dsocr`
