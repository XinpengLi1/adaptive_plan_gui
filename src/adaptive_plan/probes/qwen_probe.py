"""Local Qwen2.5-VL probe via transformers."""
from __future__ import annotations

from typing import Optional

from .base import ProbeModel, ProbeRequest


class QwenProbe(ProbeModel):
    def __init__(self, cfg: dict, cache_dir: Optional[str] = None, image_max_side: int = 1024):
        super().__init__(cache_dir, image_max_side)
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info  # noqa: F401
        except ImportError as e:
            raise SystemExit("pip install torch transformers qwen-vl-utils") from e
        self.model_name = cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.name = f"qwen:{self.model_name}"
        self.device = cfg.get("device", "cuda")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _call(self, req: ProbeRequest) -> tuple[str, Optional[int]]:
        import torch
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": p, "max_pixels": self.image_max_side**2} for p in req.images]
        content.append({"type": "text", "text": req.prompt})
        messages = [{"role": "system", "content": req.system}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = self.processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(
            self.model.device
        )
        do_sample = req.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=do_sample,
                temperature=req.temperature if do_sample else None,
            )
        gen = out[:, inputs.input_ids.shape[1] :]
        txt = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return txt, int(gen.shape[1])
