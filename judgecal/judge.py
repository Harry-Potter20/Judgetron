"""VLM judges that emit a success logit per episode.

The judge is asked a yes/no question and we read the logits of the "Yes" and
"No" tokens at the answer position. logit = l_yes - l_no is a proper binary
logit, so P(success) = sigmoid(logit). This gives a continuous score for
calibration instead of parsing free text.

Training uses BCE on that same logit. That is equivalent to cross-entropy over
the vocabulary restricted to {Yes, No}, and keeps zero-shot and fine-tuned
judges on exactly the same scoring path.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .data import Episode

PROMPT = (
    "You are verifying a robot's work. The images show the scene {view_desc}.\n"
    "Task instruction: \"{instruction}\"\n"
    "Did the robot successfully complete the task? Answer Yes or No."
)


def view_description(n: int) -> str:
    if n == 1:
        return "after execution"
    if n == 2:
        return "before (first image) and after (second image) execution"
    return f"at {n} moments in time, in order"


class Judge(Protocol):
    name: str

    def logits(self, episodes: Sequence[Episode]) -> np.ndarray: ...


class DummyJudge:
    """Deterministic fake judge for pipeline tests. Reads a latent score from meta."""

    name = "dummy"

    def logits(self, episodes):
        return np.array([ep.meta["latent"] for ep in episodes], dtype=float)


class QwenVLJudge:
    """Qwen2.5-VL judge. Requires torch, transformers>=4.49, peft, pillow.

    Memory notes: Qwen2.5-VL-3B in fp16 needs about 8 GB for inference.
    For LoRA training on a 16 GB T4, pass load_in_4bit=True (needs bitsandbytes).
    max_pixels caps visual tokens per image; 256*28*28 gives about 256 tokens each.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        adapter: str | None = None,
        max_pixels: int = 256 * 28 * 28,
        load_in_4bit: bool = False,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.name = model_id.split("/")[-1] + ("+lora" if adapter else "")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        kw = {"torch_dtype": dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4"
            )
            kw["device_map"] = {"": 0}
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw)
        if not load_in_4bit:
            self.model.to(self.device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"  # answer position is always index -1
        tok = self.processor.tokenizer
        yes, no = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError(f"'Yes'/'No' are not single tokens for {model_id}: {yes}, {no}")
        self.yes_id, self.no_id = yes[0], no[0]

    def _batch(self, episodes: Sequence[Episode]):
        from PIL import Image

        texts, images = [], []
        for ep in episodes:
            imgs = [Image.open(p).convert("RGB") for p in ep.images]
            content = [{"type": "image"} for _ in imgs]
            content.append(
                {"type": "text", "text": PROMPT.format(
                    view_desc=view_description(len(imgs)), instruction=ep.instruction)}
            )
            msgs = [{"role": "user", "content": content}]
            texts.append(self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
            images.extend(imgs)
        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def forward_logits(self, episodes: Sequence[Episode]):
        """Differentiable success logits, shape (B,)."""
        out = self.model(**self._batch(episodes))
        last = out.logits[:, -1, :].float()
        return last[:, self.yes_id] - last[:, self.no_id]

    def logits(self, episodes: Sequence[Episode], batch_size: int = 4) -> np.ndarray:
        self.model.eval()
        res = []
        with self.torch.no_grad():
            for i in range(0, len(episodes), batch_size):
                res.append(self.forward_logits(episodes[i : i + batch_size]).cpu().numpy())
        return np.concatenate(res) if res else np.zeros(0)
