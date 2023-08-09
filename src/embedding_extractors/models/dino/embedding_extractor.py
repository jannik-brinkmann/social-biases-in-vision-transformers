import argparse
import json
import math
import numpy as np
import os
from pathlib import Path
from PIL import Image
import sys
from typing import List

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

from .utils import load_pretrained_weights
from .vision_transformer import vit_base

from ...embedding_extractor import EmbeddingExtractor


class DINOEmbeddingExtractor(EmbeddingExtractor):

    def __init__(self, model_name_or_path, use_mean_pooling, extraction_layer, *args, **kwargs):
        super().__init__(model_name_or_path, extraction_layer)

        model_name_or_path = ""  # add DINO checkpoint location here
        
        self.use_mean_pooling = use_mean_pooling
        
        self.moco = self.build_model(model_name_or_path, 'vit_base', 16, 1, True)

        self.transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def build_model(self, model_name_or_path, architecture, patch_size, n_last_blocks, avgpool_patchtokens):
        model = vit_base(patch_size=patch_size, num_classes=0)

        model.cuda()
        load_pretrained_weights(model, model_name_or_path, 'student', 'vit_base', 16)
        model.eval()
        return model

    def extract_features(self, images):

        image_tensors = []
        for i in images:
            image_tensors.append(self.transform(i))
        image_tensors = torch.stack(image_tensors, dim=0).to('cuda')

        return image_tensors

    def __call__(self, images: List[Image.Image]):
        
        features = self.extract_features(images)

        with torch.no_grad():

            intermediate_output = self.moco.get_intermediate_layers(features, 1)

            # use this code for the per-layer analysis
            # intermediate_output = self.moco.get_intermediate_layers(features, 12)
            # intermediate_output = intermediate_output[self.extraction_layer - 1]  

            if self.use_mean_pooling: 

                # exclude [CLS] token from last hidden state
                hidden_states = torch.cat([x[:, 1:, :] for x in intermediate_output], dim=-1)

                # average pooling across the sequence dimension
                embeddings = torch.mean(hidden_states, 1)

            else:

                # extract [CLS] token from last hidden state
                embeddings = torch.cat([x[:, 0, :] for x in intermediate_output], dim=-1)
            
        return embeddings.cpu()
