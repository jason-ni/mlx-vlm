import codecs
import json
from copy import copy
from functools import partial
from json import JSONDecodeError
from typing import List, Optional

from transformers import AutoTokenizer

REPLACEMENT_CHAR = "\ufffd"


def _remove_space(x):
    if x and x[0] == " ":
        return x[1:]
    return x


class StreamingDetokenizer:
    """The streaming detokenizer interface so that we can detokenize one token at a time.

    Example usage is as follows:

        detokenizer = ...

        # Reset the tokenizer state
        detokenizer.reset()

        for token in generate(...):
            detokenizer.add_token(token.item())

            # Contains the whole text so far. Some tokens may not be included
            # since it contains whole words usually.
            detokenizer.text

            # Contains the printable segment (usually a word) since the last
            # time it was accessed
            detokenizer.last_segment

            # Contains all the tokens added so far
            detokenizer.tokens

        # Make sure that we detokenize any remaining tokens
        detokenizer.finalize()

        # Now detokenizer.text should match tokenizer.decode(detokenizer.tokens)
    """

    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        raise NotImplementedError()

    def add_token(self, token, skip_special_token_ids: List[int] = []):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()

    @property
    def last_segment(self):
        """Return the last segment of readable text since last time this property was accessed."""
        text = self.text
        if text and text[-1] != REPLACEMENT_CHAR:
            segment = text[self.offset :]
            self.offset = len(text)
            return segment
        return ""


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """NaiveStreamingDetokenizer relies on the underlying tokenizer
    implementation and should work with every tokenizer.

    Its complexity is O(T^2) where T is the longest line since it will
    repeatedly detokenize the same tokens until a new line is generated.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        self.reset()

    def __copy__(self):
        return type(self)(self._tokenizer)

    def reset(self):
        self.offset = 0
        self._tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token, skip_special_token_ids: List[int] = []):
        if token in skip_special_token_ids:
            return
        self._current_tokens.append(token)

    def finalize(self):
        self._tokens.extend(self._current_tokens)
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
        if self._current_text and self._current_text[-1] == "\n":
            self._tokens.extend(self._current_tokens)
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text

    @property
    def tokens(self):
        return self._tokens


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.

    Handles UTF-8 byte tokens (like <0xE5><0xA4><0xA2>) by accumulating them
    and decoding as UTF-8 bytes, which is necessary for multi-byte characters
    like Chinese that may not be in the vocabulary.
    """

    def __init__(self, tokenizer, trim_space=True):
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        self.is_byte_token = [False] * len(tokenizer.vocab)
        self.byte_value = [0] * len(tokenizer.vocab)

        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value
            # Mark byte tokens and store their byte value
            if value.startswith("<0x") and len(value) >= 6 and value[5] == ">":
                self.is_byte_token[tokenid] = True
                self.byte_value[tokenid] = int(value[3:5], 16)

        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self._byte_buffer = bytearray()
        self.text = ""
        self.tokens = []

    def _flush_bytes(self):
        """Decode accumulated bytes as UTF-8 and append to unflushed text."""
        if self._byte_buffer:
            try:
                decoded = self._byte_buffer.decode("utf-8")
                self._unflushed += decoded
            except UnicodeDecodeError:
                # If decoding fails, use replacement character
                self._unflushed += self._byte_buffer.decode("utf-8", errors="replace")
            self._byte_buffer = bytearray()

    def add_token(self, token, skip_special_token_ids: List[int] = []):
        if token in skip_special_token_ids:
            return

        if self.is_byte_token[token]:
            # Accumulate byte tokens
            self._byte_buffer.append(self.byte_value[token])
            return

        # Flush any accumulated bytes before processing regular token
        self._flush_bytes()

        v = self.tokenmap[token]
        if v and v[0] == "\u2581":
            if self.text or not self.trim_space:
                self.text += self._unflushed.replace("\u2581", " ")
            else:
                self.text = _remove_space(self._unflushed.replace("\u2581", " "))
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        # Flush any remaining bytes
        self._flush_bytes()

        if self.text or not self.trim_space:
            self.text += self._unflushed.replace("\u2581", " ")
        else:
            self.text = _remove_space(self._unflushed.replace("\u2581", " "))
        self._unflushed = ""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for OpenAI style BPE models.

    It adds tokens to the text if the next token starts with a space similar to
    the SPM detokenizer.
    """

    _byte_decoder = None

    def __init__(self, tokenizer, trim_space=False):
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

        # Make the BPE byte decoder from
        # https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.make_byte_decoder()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token, skip_special_token_ids: List[int] = []):
        if token in skip_special_token_ids:
            return
        v = self.tokenmap[token]
        # if the token starts with space
        try:
            starts_with_space = self._byte_decoder[v[0]] == 32
        except (KeyError, IndexError, TypeError):
            self._unflushed += v
            return
        if starts_with_space:
            try:
                current_text = bytearray(
                    self._byte_decoder[c] for c in self._unflushed
                ).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                current_text = self._unflushed
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += _remove_space(current_text)
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        try:
            current_text = bytearray(
                self._byte_decoder[c] for c in self._unflushed
            ).decode("utf-8", errors="ignore")
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += _remove_space(current_text)
        except (UnicodeDecodeError, KeyError):
            print(f"Warning: could not decode bytes: {self._unflushed}")
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        """See https://github.com/openai/gpt-2/blob/master/src/encoder.py for the rationale."""
        if cls._byte_decoder is not None:
            return

        char_to_bytes = {}
        limits = [
            0,
            ord("!"),
            ord("~") + 1,
            ord("¡"),
            ord("¬") + 1,
            ord("®"),
            ord("ÿ") + 1,
        ]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


class _ServerTokenStreamer:
    """Emit server text deltas per token while buffering incomplete UTF-8."""

    def __init__(self, tokenizer, detokenizer):
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer
        self._decoder = None
        self._started = False
        self._tokenmap = None
        self._mode = "delegate"
        self._trim_space = False

        if isinstance(detokenizer, SPMStreamingDetokenizer):
            self._mode = "spm"
            self._tokenmap = detokenizer.tokenmap
            self._trim_space = detokenizer.trim_space
            self._is_byte_token = detokenizer.is_byte_token
            self._byte_value = detokenizer.byte_value
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        elif isinstance(detokenizer, BPEStreamingDetokenizer):
            self._mode = "bpe"
            self._tokenmap = detokenizer.tokenmap
            self._trim_space = detokenizer.trim_space
            self._byte_decoder = detokenizer._byte_decoder
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def _emit_text(self, text: str) -> str:
        if text and not self._started and self._trim_space and text[0] == " ":
            text = text[1:]
        if text:
            self._started = True
        return text

    def _fallback_decode(self, token: int) -> str:
        try:
            return self._tokenizer.decode([token])
        except Exception:
            return ""

    def _spm_text_for_token(self, token: int) -> str:
        try:
            if self._is_byte_token[token]:
                byte_value = self._byte_value[token]
                return self._decoder.decode(bytes((byte_value,)), final=False)
            value = self._tokenmap[token]
        except (IndexError, TypeError):
            return self._fallback_decode(token)

        if value is None:
            return self._fallback_decode(token)
        return value.replace("\u2581", " ")

    def _bpe_text_for_token(self, token: int) -> str:
        try:
            value = self._tokenmap[token]
        except (IndexError, TypeError):
            return self._fallback_decode(token)

        if value is None:
            return self._fallback_decode(token)

        try:
            token_bytes = bytes(self._byte_decoder[c] for c in value)
        except KeyError:
            return self._fallback_decode(token)
        return self._decoder.decode(token_bytes, final=False)

    def _decode_token(self, token: int) -> str:
        if self._mode == "spm":
            return self._emit_text(self._spm_text_for_token(token))
        if self._mode == "bpe":
            return self._emit_text(self._bpe_text_for_token(token))
        return self._fallback_advance(token, finish_reason=None)

    def _fallback_advance(self, token: int, finish_reason: Optional[str]) -> str:
        if finish_reason == "stop":
            self._detokenizer.finalize()
        else:
            self._detokenizer.add_token(token)
            if finish_reason is not None:
                self._detokenizer.finalize()
        return self._detokenizer.last_segment

    def advance(self, token: int, finish_reason: Optional[str]) -> str:
        if self._mode == "delegate":
            return self._fallback_advance(token, finish_reason)

        parts = []
        if finish_reason != "stop":
            parts.append(self._decode_token(token))
        if finish_reason is not None:
            parts.append(self._emit_text(self._decoder.decode(b"", final=True)))
        return "".join(parts)

    def finalize(self) -> str:
        if self._mode == "delegate":
            self._detokenizer.finalize()
            return self._detokenizer.last_segment
        return self._emit_text(self._decoder.decode(b"", final=True))


class TokenizerWrapper:
    """A wrapper that combines an HF tokenizer and a detokenizer.

    Accessing any attribute other than the ``detokenizer`` is forwarded to the
    huggingface tokenizer.
    """

    def __init__(self, tokenizer, detokenizer_class=NaiveStreamingDetokenizer):
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer_class(tokenizer)

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self._detokenizer
        else:
            return getattr(self._tokenizer, attr)


def make_streaming_detokenizer(processor):
    """Return an isolated, reset streaming detokenizer for a processor."""
    detokenizer = copy(processor.detokenizer)
    detokenizer.reset()
    return detokenizer


def _match(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))

    return a == b


def _is_spm_decoder(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder):
    return isinstance(decoder, dict) and decoder.get("type", None) == "ByteLevel"


def _vocab_uses_byte_level_bpe(vocab):
    """Check whether vocab tokens use GPT-2 byte-level BPE encoding (Ġ space markers).

    Some tokenizers (e.g. DeepSeek) carry an SPM-style decoder config but were
    trained with byte-level BPE, so their vocab tokens contain ``Ġ`` (U+0120)
    instead of ``▁`` (U+2581) for spaces.  The SPM streaming detokenizer cannot
    handle these, so we detect the mismatch and fall back to the BPE one.
    """
    if not vocab:
        return False
    g_count = 0
    checked = 0
    for token_str in vocab:
        # Skip special tokens like <|det|>, <｜begin▁of▁sentence｜>, etc.
        if len(token_str) >= 2 and token_str[0] == "<" and token_str[-1] == ">":
            continue
        checked += 1
        if token_str.startswith("\u0120"):  # Ġ
            g_count += 1
        if checked >= 2000:
            break
    return checked > 0 and g_count > checked * 0.05


def load_tokenizer(model_path, return_tokenizer=True, tokenizer_config_extra={}):
    """Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.
    """
    detokenizer_class = NaiveStreamingDetokenizer

    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        with open(tokenizer_file, "r") as f:
            try:
                tokenizer_content = json.load(f)
            except JSONDecodeError as e:
                raise JSONDecodeError("Failed to parse tokenizer.json", e.doc, e.pos)

        spm_selected = False
        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
                spm_selected = True
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
                spm_selected = True
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

        # Some tokenizers (e.g. DeepSeek OCR) have an SPM-style decoder config but
        # a byte-level BPE vocab (Ġ-encoded).  Override to BPE in that case.
        if spm_selected:
            vocab = tokenizer_content.get("model", {}).get("vocab", {})
            if _vocab_uses_byte_level_bpe(vocab):
                detokenizer_class = BPEStreamingDetokenizer

    if return_tokenizer:
        return TokenizerWrapper(
            AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra),
            detokenizer_class,
        )
    else:
        return detokenizer_class
