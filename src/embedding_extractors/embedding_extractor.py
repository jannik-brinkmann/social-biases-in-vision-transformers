import torch

from collections import OrderedDict
from typing import List

from PIL import Image
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel


EXTRACTION_LAYER = OrderedDict(
    [   
        # BeitForMaskedImageModeling
        ('microsoft/beit-base-patch16-224-pt22k', 9),
        ('microsoft/beit-large-patch16-224-pt22k', 14),

        # BeitForImageClassification
        ('microsoft/beit-base-patch16-224-pt22k-ft22k', 12),
        ('microsoft/beit-large-patch16-224-pt22k-ft22k', 24),
        ('microsoft/beit-base-patch16-384', 12),
        ('microsoft/beit-large-patch16-224', 22),
        ('microsoft/beit-large-patch16-384', 22),
        ('microsoft/beit-large-patch16-512', 22),

        # ImageGPTForCausalImageModeling
        ('openai/imagegpt-small', 11),
        ('openai/imagegpt-medium', 14),
        ('openai/imagegpt-large', 20),
    ]
)


class EmbeddingExtractor:

    def __init__(self, model_name_or_path, extraction_layer = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name_or_path = model_name_or_path

        self.embd_dim = 0
        if extraction_layer == None:
            self.extraction_layer = self._get_extraction_layer()
        else:
            self.extraction_layer = int(extraction_layer)

    def __call__(self, images: List[Image.Image], *args, **kwargs):
        raise NotImplementedError

    def _get_extraction_layer(self):

        output = None

        if self.model_name_or_path in EXTRACTION_LAYER.keys():
            output = EXTRACTION_LAYER[self.model_name_or_path]

        return output
