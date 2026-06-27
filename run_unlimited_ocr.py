import mlx.core as mx
from mlx_vlm import load, generate


def make_no_repeat_ngram(ngram_size=35, window=128, whitelist=frozenset({128815})):
    """Match PT's SlidingWindowNoRepeatNgramProcessor.
    Only bans the token that would complete a previously seen n-gram
    whose prefix matches the current context."""
    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if tokens.size < ngram_size:
            return logits
        seq = tokens.tolist()
        search_start = max(0, len(seq) - window)
        search_end = len(seq) - ngram_size + 1
        if search_end <= search_start:
            return logits
        current_prefix = tuple(seq[-(ngram_size - 1):])
        banned = set()
        for i in range(search_start, search_end):
            ngram = seq[i:i + ngram_size]
            if tuple(ngram[:-1]) == current_prefix:
                banned.add(ngram[-1])
        banned.difference_update(whitelist)
        for tid in banned:
            logits[:, tid] = -float("inf")
        return logits
    return processor


model, processor = load("/Volumes/Realtek/models/mlx/unlimited-ocr-hybrid-mlx")

response = generate(model, processor,
                    prompt="<image> document parsing.",
                    image="examples/images/paper_test.jpg",
                    max_tokens=8192, verbose=True, temperature=0,
                    logits_processors=[make_no_repeat_ngram()])

stop_str = "<｜end▁of▁sentence｜>"
output = response.text
if output.endswith(stop_str):
    output = output[:-len(stop_str)]
print("\n\n=== OUTPUT ===")
print(output.strip())
print(f"\n{response.generation_tokens} tokens generated")
