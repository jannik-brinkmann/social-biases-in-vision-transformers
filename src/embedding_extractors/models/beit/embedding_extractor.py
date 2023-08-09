import logging
import torch

from collections import OrderedDict
from PIL import Image
from typing import List
from transformers import AutoConfig, AutoFeatureExtractor, BeitForMaskedImageModeling

from ...embedding_extractor import EmbeddingExtractor


logger = logging.getLogger(__name__)


class BeitEmbeddingExtractor(EmbeddingExtractor):

    def __init__(self, model_name_or_path, use_mean_pooling, extraction_layer, *args, **kwargs):
        super().__init__(model_name_or_path, extraction_layer)

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.use_mean_pooling = use_mean_pooling

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

        self.beit = BeitForMaskedImageModeling.from_pretrained(self.model_name_or_path, config=self.config)
        self.beit.to(self.device)
        self.beit.eval()

        self.embed_dim = self.config.hidden_size

    def __call__(self, images: List[Image.Image]) -> torch.Tensor:
        features = self.feature_extractor(images, return_tensors="pt")
        features.to(self.device)

        with torch.no_grad():

            outputs = self.beit(**features, output_hidden_states=True, return_dict=True)

            embeddings = self.extract_embeddings(outputs, self.extraction_layer)

        return embeddings.cpu()

    def extract_embeddings(self, outputs, extraction_layer: int) -> torch.Tensor:

        if self.config.use_mean_pooling: 
               
            # extract l-th hidden state $h^l$ (excluding [CLS] token)
            h_l = outputs.hidden_states[extraction_layer][:, 1:, :]

            # compute $ n^l = layer\_norm(h^l) $
            if extraction_layer == None or extraction_layer == 11:
                n_l = self.beit.layernorm(h_l)
            else:
                n_l = self.beit.beit.encoder.layer[extraction_layer + 1].layernorm_before(h_l)

            # average pool $n^l$ across the sequence dimension
            embeddings = torch.mean(n_l, 1)

        else:
                
            # extract [CLS] token from l-th hidden state
            h_l = outputs.hidden_states[extraction_layer][:, 0, :]

            # compute $ n^l = layer\_norm(h^l) $
            embeddings = self.beit.beit.encoder.layer[extraction_layer + 1].layernorm_before(h_l)

        return embeddings
