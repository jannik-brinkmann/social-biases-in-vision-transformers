import math
import numpy as np

import torch

from PIL import Image
from transformers import AutoConfig, AutoFeatureExtractor, ImageGPTForCausalImageModeling
from typing import List

from ...embedding_extractor import EmbeddingExtractor


class ImageGPTEmbeddingExtractor(EmbeddingExtractor):

    def __init__(self, model_name_or_path, use_mean_pooling, extraction_layer, *args, **kwargs):
        super().__init__(model_name_or_path, extraction_layer)
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)  # ImageGPT does not use a [CLS] token
        self.config.use_mean_pooling = use_mean_pooling

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

        self.imagegpt = ImageGPTForCausalImageModeling.from_pretrained(model_name_or_path)
        self.imagegpt.to(self.device)
        self.imagegpt.eval()

    def __call__(self, images: List[Image.Image]):

        embeddings = None

        # due to resource limitations and size of ImageGPT-L, we use small batch sizes on ImageGPT
        batch_size = 5
        for i in np.arange(0, len(images), batch_size).tolist():

            image_batch = images[i : i + batch_size]

            features = self.feature_extractor(image_batch, return_tensors="pt")
            features.to(self.device)

            with torch.no_grad():

                # extract hidden states of images
                outputs = self.imagegpt(**features, output_hidden_states=True, return_dict=True)

                # extract l-th hidden state $h^l$
                h_l = outputs.hidden_states[self.extraction_layer]

                # compute $ n^l = layer\_norm(h^l) $
                n_l = self.imagegpt.transformer.h[self.extraction_layer + 1].ln_1(h_l)

                # average pool $n^l$ across the sequence dimension
                if not torch.is_tensor(embeddings):
                    embeddings = torch.mean(n_l, 1)
                else:
                    embeddings = torch.cat((embeddings, torch.mean(n_l, 1)), 0)

        return embeddings.cpu()
