from mlx_vlm import load, generate

model, processor = load("/Volumes/Realtek/models/mlx/unlimited-ocr-8bit-mlx")

# Single-image OCR (Gundam mode)
response = generate(model, processor,
                    prompt="<image> document parsing.",
                    image="examples/images/paper_test.png",
                    max_tokens=8192, verbose=True)

