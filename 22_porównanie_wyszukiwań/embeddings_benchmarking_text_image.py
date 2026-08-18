import os
import platform
import subprocess
import torch
import open_clip
from qdrant_client import QdrantClient

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "xlm-roberta-base-ViT-B-32"
pretrained = "laion5b_s13b_b90k"

print(f"Ładowanie modelu {model_name} na urządzeniu: {device}...")
model, _, _ = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)