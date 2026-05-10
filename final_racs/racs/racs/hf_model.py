"""
HuggingFace Transformers backend for RACS.

A single local LLM handles three jobs used across the offline and online phases:

  * `embed(texts)`              — mean-pooled last-hidden-state embeddings for items
  * `generate(prompt)`          — greedy generation for risk scoring
  * `contrast_logprob(prompt,   — softmax over two first-token IDs.
                     word_a, word_b)  Used by ACS for the spec's
                                      exp(logp(ENGAGED)) / [exp(logp(ENGAGED))
                                                         + exp(logp(SKIPPED))].

No network calls at inference time. Weights are downloaded once on first run
(via HuggingFace Hub) and cached to `~/.cache/huggingface/hub/`. Gated models
(Llama, Gemma) require a one-time license acceptance + `huggingface-cli login`.
"""

import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

HF_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"


class HFLLM:
    """Local causal-LM wrapper. Handles embed, generate, and two-token contrast."""

    def __init__(self,
                 model_name: str = HF_MODEL_DEFAULT,
                 device: str = "cuda",
                 dtype: str = "float16",
                 load_in_4bit: bool = False,
                 attn_impl: str = "sdpa"):
        self.model_name = model_name

        logger.info("Loading HF model %s (dtype=%s, 4bit=%s, attn=%s)",
                    model_name, dtype, load_in_4bit, attn_impl)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right-padding lets us index the last real-token logit per row as
        # `attention_mask.sum(-1) - 1`. Required by `contrast_logprob_batch`.
        self.tokenizer.padding_side = "right"

        kwargs: dict = {"attn_implementation": attn_impl}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            # 4-bit quantization requires accelerate + device_map
            kwargs["device_map"] = device
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            dtype_map = {
                "float16":  torch.float16,
                "bfloat16": torch.bfloat16,
                "float32":  torch.float32,
            }
            if dtype not in dtype_map:
                raise ValueError(f"Unknown dtype {dtype!r}; expected one of {list(dtype_map)}")
            # `torch_dtype=` is accepted on all transformers versions; the new
            # `dtype=` alias is silently ignored on <4.45 → loads fp32. Stick
            # with torch_dtype for portability across transformers versions.
            kwargs["torch_dtype"] = dtype_map[dtype]

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except (ValueError, ImportError) as e:
            # flash_attention_2 may not be installed; fall back to sdpa
            if attn_impl == "flash_attention_2":
                logger.warning("flash_attention_2 unavailable (%s) — falling "
                               "back to sdpa", e)
                kwargs["attn_implementation"] = "sdpa"
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            else:
                raise
        if not load_in_4bit:
            self.model = self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        logger.info("HF model ready on %s", self.device)

        # cache first-token IDs for ACS word pairs
        self._token_cache: dict[str, int] = {}

    # ---------------------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, texts: list[str],
              batch_size: int = 8,
              max_length: int = 512) -> np.ndarray:
        """
        Mean-pooled last-hidden-state embedding for each input text.
        Returns an (N, D) float32 array. Pooled over valid tokens (attention mask).
        """
        all_emb: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            out = self.model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[-1]                   # (B, T, D)
            mask = inputs.attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_emb.append(pooled.float().cpu().numpy())
        return np.vstack(all_emb)

    # ---------------------------------------------------------------------
    # Generation (for risk scoring)
    # ---------------------------------------------------------------------

    def _wrap_prompt(self, user_message: str) -> str:
        """Apply the model's chat template when available (instruct models)."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_message

    @torch.no_grad()
    def generate(self, user_message: str,
                 max_new_tokens: int = 32,
                 temperature: float = 0.0) -> str:
        """Greedy generation at temperature=0 (or sampled if temperature>0)."""
        prompt = self._wrap_prompt(user_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    # ---------------------------------------------------------------------
    # Contrastive logprob (for ACS)
    # ---------------------------------------------------------------------

    def _first_token_id(self, word: str) -> int:
        """Cached: first token of ' ' + word (leading space = mid-sentence token)."""
        key = word
        if key in self._token_cache:
            return self._token_cache[key]
        ids = self.tokenizer.encode(" " + word, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"Tokenizer produced no tokens for {word!r}")
        self._token_cache[key] = ids[0]
        return ids[0]

    @torch.no_grad()
    def contrast_logprob(self, user_message: str,
                         word_a: str, word_b: str) -> float:
        """
        P_A = exp(logp(word_a)) / [exp(logp(word_a)) + exp(logp(word_b))]

        Computed at the next-token position after the chat-templated prompt.
        The two words must map to distinct first-token IDs (raises otherwise).
        """
        id_a = self._first_token_id(word_a)
        id_b = self._first_token_id(word_b)
        if id_a == id_b:
            raise RuntimeError(
                f"First-token collision: {word_a!r} and {word_b!r} "
                f"both start with token id {id_a}. Pick a different contrast pair."
            )

        prompt = self._wrap_prompt(user_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        last_logits = out.logits[0, -1, :]
        l_a = last_logits[id_a].item()
        l_b = last_logits[id_b].item()
        m = max(l_a, l_b)
        ea = np.exp(l_a - m)
        eb = np.exp(l_b - m)
        return float(ea / (ea + eb))

    @torch.no_grad()
    def contrast_logprob_batch(self, user_messages: list[str],
                               word_a: str, word_b: str,
                               batch_size: int = 16,
                               max_length: int = 1024) -> list[float]:
        """Batched version of `contrast_logprob`.

        Tokenises all prompts together with right-padding, runs ONE forward
        pass per micro-batch, and reads the (id_a, id_b) logits at the last
        non-pad position of each row. Mathematically identical to calling
        `contrast_logprob` per prompt but ~5–10× faster on GPU.
        """
        if not user_messages:
            return []

        id_a = self._first_token_id(word_a)
        id_b = self._first_token_id(word_b)
        if id_a == id_b:
            raise RuntimeError(
                f"First-token collision: {word_a!r} and {word_b!r} "
                f"both start with token id {id_a}. Pick a different contrast pair.")

        results: list[float] = []
        prompts = [self._wrap_prompt(m) for m in user_messages]

        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            inputs = self.tokenizer(
                chunk, return_tensors="pt",
                padding=True, truncation=True, max_length=max_length,
            ).to(self.device)
            out = self.model(**inputs)
            attn = inputs.attention_mask                       # (B, L)
            last_idx = attn.sum(dim=1) - 1                      # (B,)
            batch_idx = torch.arange(out.logits.size(0), device=out.logits.device)
            last_logits = out.logits[batch_idx, last_idx]       # (B, V)
            l_a = last_logits[:, id_a]
            l_b = last_logits[:, id_b]
            m = torch.maximum(l_a, l_b)
            ea = torch.exp(l_a - m)
            eb = torch.exp(l_b - m)
            probs = (ea / (ea + eb)).float().cpu().tolist()
            results.extend(probs)
        return results
