import numpy as np
import torch

from PIL import Image
from typing import List
from transformers import AutoConfig, ViTFeatureExtractor, ViTMSNModel

from ...embedding_extractor import EmbeddingExtractor


class ViTMSNEmbeddingExtractor(EmbeddingExtractor):

    def __init__(self, model_name_or_path, use_mean_pooling, extraction_layer, *args, **kwargs):
        super().__init__(model_name_or_path, extraction_layer)
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.use_mean_pooling = use_mean_pooling

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

        self.model = ViTMSNModel.from_pretrained(self.model_name_or_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()

        self.embed_dim = self.config.hidden_size

    def __call__(self, images: List[Image.Image], *args, **kwargs):
        features = self.feature_extractor(images, return_tensors="pt")
        features.to(self.device)

        with torch.no_grad():

            # compute embedding using the encoder
            outputs = self.model(**features, output_hidden_states=True, return_dict=True)

            embeddings = self.extract_embeddings(outputs, self.extraction_layer)

        return embeddings.cpu()

    def extract_embeddings(self, outputs, extraction_layer) -> torch.Tensor:

        if self.config.use_mean_pooling: 

            if extraction_layer == None or extraction_layer == 11:

                # exclude [CLS] token from last hidden state
                hidden_states = outputs.last_hidden_state[:, 1:, :]

                # average pooling across the sequence dimension
                embeddings = torch.mean(hidden_states, 1)

            else:
                
                # exclude [CLS] token from hidden state
                hidden_states = outputs.hidden_states[extraction_layer][:, 1:, :]

                # average pooling across the sequence dimension
                embeddings = torch.mean(hidden_states, 1)

        else:

            if extraction_layer == None or extraction_layer == 11:

                # extract [CLS] token from last hidden state
                embeddings = outputs.last_hidden_state[:, 0, :]

            else: 

                # exclude [CLS] token from hidden state
                hidden_states = outputs.hidden_states[extraction_layer][:, 0, :]

                # average pooling across the sequence dimension
                embeddings = torch.mean(hidden_states, 1)

        return embeddings

