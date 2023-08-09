import importlib

from collections import OrderedDict

from transformers import (
    AutoConfig,
    BeitConfig,
    ChineseCLIPConfig,
    CLIPConfig,
    ImageGPTConfig,
    ViTConfig,
    ViTMAEConfig,
    ViTMSNConfig
)

from ..beit import BeitEmbeddingExtractor
from ..dino import DINOEmbeddingExtractor
from ..imagegpt import ImageGPTEmbeddingExtractor
from ..vit import ViTEmbeddingExtractor
from ..moco_v3 import ViTMoCoEmbeddingExtractor
from ..vit_msn import ViTMSNEmbeddingExtractor


EMBEDDING_EXTRACTOR_MAPPING = OrderedDict(
    [
        (BeitConfig, BeitEmbeddingExtractor),
        (ImageGPTConfig, ImageGPTEmbeddingExtractor),
        (ViTConfig, ViTEmbeddingExtractor),
        (ViTMAEConfig, ViTEmbeddingExtractor),
        ('facebook/vit-moco', ViTMoCoEmbeddingExtractor),  # ViT-MoCo is not supported by HuggingFace
        (ViTMSNConfig, ViTMSNEmbeddingExtractor),
        ('facebook/dino-vitb16', ViTEmbeddingExtractor), 
    ]
)


class AutoEmbeddingExtractor:

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f'{self.__class__.__name__} is designed to be instantiated using .from_pretrained() method.'
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path, use_mean_pooling, extraction_layer, *args, **kwargs):
        
        if model_name_or_path in EMBEDDING_EXTRACTOR_MAPPING.keys():

            embedding_extractor = EMBEDDING_EXTRACTOR_MAPPING[model_name_or_path]
        
        else:

            config = AutoConfig.from_pretrained(model_name_or_path)
            if type(config) in EMBEDDING_EXTRACTOR_MAPPING.keys():
                embedding_extractor = EMBEDDING_EXTRACTOR_MAPPING[type(config)]
            else:
                raise EnvironmentError(
                    f'{cls.__class__.__name__} is designed to be instantiated given a registered ModelConfig.'
                )

        return embedding_extractor(model_name_or_path, use_mean_pooling, extraction_layer)
