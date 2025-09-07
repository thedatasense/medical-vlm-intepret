#!/usr/bin/env python3
"""
Official-style LLaVA-Rad demo runner

This mirrors the usage from microsoft/llava-rad docs using the llava
library's load_pretrained_model and conversation templates.

Usage:
  python llava_rad_official_demo.py \
    --image "https://openi.nlm.nih.gov/imgs/512/253/253/CXR253_IM-1045-1001.png" \
    --prompt "Describe the findings of the chest x-ray."

Notes:
- Requires installing LLaVA-Rad (or LLaVA providing `llava.*` modules):
    git clone https://github.com/microsoft/llava-rad.git
    cd llava-rad && pip install --upgrade pip && pip install -e .
"""

from __future__ import annotations

import argparse
from io import BytesIO
from typing import Optional

import requests
import torch
from PIL import Image


def load_image(image_file: str) -> Image.Image:
    if image_file.startswith("http://") or image_file.startswith("https://"):
        resp = requests.get(image_file, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description="Run LLaVA-Rad with an image and prompt")
    ap.add_argument("--image", required=True, help="Image URL or local path")
    ap.add_argument(
        "--prompt",
        default="Describe the findings of the chest x-ray.",
        help="User prompt/question"
    )
    ap.add_argument(
        "--model-path",
        default="microsoft/llava-rad",
        help="Model ID or path"
    )
    ap.add_argument(
        "--model-base",
        default="lmsys/vicuna-7b-v1.5",
        help="Base language model for LLaVA-Rad"
    )
    ap.add_argument(
        "--conv-mode",
        default="v1",
        help="Conversation template mode (e.g., v1)"
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate"
    )
    args = ap.parse_args()

    # Lazy import to ensure the package is installed
    from llava.utils import disable_torch_init
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX

    disable_torch_init()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name="llavarad",
    )

    image = load_image(args.image)
    pixel_values = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ]

    # dtype/device handling
    if device == "cuda":
        pixel_values = pixel_values.half().to(device)
    else:
        pixel_values = pixel_values.float()

    query = f"<image>\n{args.prompt}\n"

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    if device == "cuda":
        input_ids = input_ids.to(device)

    stopping_criteria = KeywordsStoppingCriteria(["</s>"], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
            images=pixel_values,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(
        output_ids[:, input_ids.shape[0] if input_ids.dim() == 1 else input_ids.shape[1] :],
        skip_special_tokens=True,
    )[0].strip()

    print(outputs)


if __name__ == "__main__":
    main()

