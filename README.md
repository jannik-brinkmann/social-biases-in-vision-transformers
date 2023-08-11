# A Multidimensional Analysis of Social Biases in Vision Transformers
[![arXiv](https://img.shields.io/badge/arXiv-2308.01948-b31b1b.svg)](https://arxiv.org/abs/2308.01948)

This is the official implementation of "A Multidimensional Analysis of Social Biases in Vision Transformers" (Brinkmann et al., 2023).

> The embedding spaces of image models have been shown to encode a range of social biases such as racism and sexism. Here, we investigate the specific factors that contribute to the emergence of these biases in Vision Transformers (ViT). Therefore, we measure the impact of training data, model architecture, and training objectives on social biases in the learned representations of ViTs. Our findings indicate that counterfactual augmentation training using diffusion-based image editing can mitigate biases, but does not eliminate them. Moreover, we find that larger models are less biased than smaller models, and that joint-embedding models are less biased than reconstruction-based models. In addition, we observe inconsistencies in the learned social biases. To our surprise, ViTs can exhibit opposite biases when trained on the same data set using different self-supervised training objectives. Our findings give insights into the factors that contribute to the emergence of social biases and suggests that we could achieve substantial fairness gains based on model design choices.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets and Models

We use [ImageNet-1k](https://image-net.org/download) for the counterfactual augmentation training and the [iEAT](https://github.com/ryansteed/ieat) dataset to measure social biases in the embeddings. 
To generate textual descriptions of each image, we use [CLIP Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator/blob/main/app.py). 
Then, we generate counterfactual descriptions using the gender terms pairs of [UCLA NLP](https://github.com/uclanlp/corefBias/blob/master/WinoBias) and use those to generate counterfactual images using [Diffusion-based Semantic Image Editing using Mask Guidance](https://arxiv.org/abs/2210.11427) (see [HuggingFace space](https://huggingface.co/spaces/nielsr/text-based-inpainting)). 

We adopt [HuggingFace](https://huggingface.co/docs/transformers/model_doc/imagegpt)'s Transformers and Ross Wightman's [Timm](https://github.com/huggingface/pytorch-image-models) to support a range of different Vision Transformers. 
The models from the [HuggingFace Hub](https://huggingface.co/models) are downloaded in the code. 
You can download the MoCo-v3 checkpoint at [MoCo-v3](https://github.com/facebookresearch/moco-v3).

## Citation
```bibtex
@article{brinkmann2023socialbiases,
    title   = {A Multidimensional Analsis of Social Biases in Vision Transformers},
    author  = {Brinkmann, Jannik and Swoboda, Paul and Bartelt, Christian},
    journal = {arXiv preprint arXiv:2308.01948},
    year    = {2023}
}
```
