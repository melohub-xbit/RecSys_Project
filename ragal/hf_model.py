import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

HF_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"

class HFLLM:
    def __init__(self,
                 model_name: str = HF_MODEL_DEFAULT,
                 device: str = "cuda",
                 dtype: str = "float16",
                 load_in_4bit: bool = False):
        self.model_name = model_name

        logger.info("Loading HF model %s (dtype=%s, 4bit=%s)",
                    model_name, dtype, load_in_4bit)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
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
            kwargs["dtype"] = dtype_map[dtype]

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_in_4bit:
            self.model = self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        logger.info("HF model ready on %s", self.device)

        self._token_cache: dict[str, int] = {}

    @torch.no_grad()
    def embed(self, texts: list[str],
              batch_size: int = 8,
              max_length: int = 512) -> np.ndarray:
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
            hidden = out.hidden_states[-1]
            mask = inputs.attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_emb.append(pooled.float().cpu().numpy())
        return np.vstack(all_emb)

    def _wrap_prompt(self, user_message: str) -> str:
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

    def _first_token_id(self, word: str) -> int:
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
