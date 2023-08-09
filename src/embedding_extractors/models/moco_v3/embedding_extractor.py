from typing import List

import torch
from PIL import Image

from ...embedding_extractor import EmbeddingExtractor
import timm
from timm.models.vision_transformer import VisionTransformer
from transformers import DeiTFeatureExtractor, DeiTModel
import numpy as np
import torch.nn as nn
import math
from functools import partial, reduce
from operator import mul
import os
import torchvision.transforms as transforms


class ViTMoCoEmbeddingExtractor(EmbeddingExtractor):

    def __init__(self, model_name_or_path, use_mean_pooling, extraction_layer, *args, **kwargs):
        super().__init__(model_name_or_path, extraction_layer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name_or_path = ""  # add ViT-MoCo checkpoint location here
        self.model_name_or_path = model_name_or_path

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        self.model = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        if os.path.isfile(model_name_or_path):
            linear_keyword = 'head'
            print("=> loading checkpoint '{}'".format(model_name_or_path))
            checkpoint = torch.load(model_name_or_path, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = self.model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
        
        self.model.eval()

        self.embed_dim = self.model.head.in_features

    def __call__(self, images: List[Image.Image], *args, **kwargs):
        features = []
        for i in images:
            features.append(self.transform(i))
        features = torch.stack(features, dim=0)

        with torch.no_grad():

            outputs = self.model._intermediate_layers(features, 12)[self.extraction_layer]
            if self.extraction_layer == None or self.extraction_layer == 11:
                hidden_state = self.model.norm(outputs)
            else:
                hidden_state = self.model.blocks[self.extraction_layer + 1].norm1(outputs)

            embeddings = hidden_state[:, 0]

        return embeddings.cpu()


        

            
