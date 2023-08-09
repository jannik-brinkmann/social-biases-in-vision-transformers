import os
import sys

import argparse
import logging
import torch
import transformers

from PIL import Image
from transformers import AutoConfig
from tqdm import tqdm

from src import AutoEmbeddingExtractor, ImageEmbeddingAssociationTest, ASSOCIATION_TESTS


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--use_mean_pooling', default=False)
    parser.add_argument('--extraction_layer', default=None)
    args = parser.parse_args()
    print(args.extraction_layer, flush=True)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # execute image embedding association test
    embedding_extractor = AutoEmbeddingExtractor.from_pretrained(
        model_name_or_path=args.model_name_or_path, 
        use_mean_pooling=args.use_mean_pooling,
        extraction_layer=args.extraction_layer
    )
    ieat = ImageEmbeddingAssociationTest(embedding_extractor)
    for n, s in ASSOCIATION_TESTS.items():

        e, p = ieat(s)
        print(f"Model: {args.model_name_or_path}, Test: {n}, Effect Size: {e}, p-Value: {p}")
