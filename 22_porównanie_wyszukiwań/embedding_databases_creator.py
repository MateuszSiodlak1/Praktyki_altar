import json
import os
import torch
import open_clip
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct