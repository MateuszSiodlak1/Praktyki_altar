import json
import os
import torch
import open_clip
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "xlm-roberta-base-ViT-B-32"
pretrained = "laion5b_s13b_b90k"

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)