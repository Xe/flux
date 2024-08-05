import argparse
from diffusers import FluxPipeline
import torch
from huggingface_hub import login, snapshot_download
import os


def main():
    token = os.environ["HUGGING_FACE_HUB_TOKEN"]
    login(token)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    pipe("a cool dog", height=1024, width=1024, num_inference_steps=28)

    pipe.save_pretrained("flux-cache/", safe_serialization=True)


if __name__ == "__main__":
    main()
