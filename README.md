# BindScreen

BindScreen: Protein-Centric Contrastive Learning for Sequence-Based Virtual Screening

\[[Dataset on HuggingFace](https://huggingface.co/datasets/SaeedLab/BindScreen)\] | \[[Model Collection](https://huggingface.co/collections/SaeedLab/bindscreen)\] | \[[Cite](#citation)\]

## Abstract
Virtual screening aims to identify candidate molecules that bind to a target protein, playing a central role in computational drug discovery. Sequence-based deep learning methods offer a more broadly applicable alternative to structure-based approaches, since they do not require 3D structural information. However, they typically require a separate forward pass per protein-molecule pair, limiting their scalability to large molecular libraries. Contrastive learning methods inspired by CLIP address this by encoding proteins and molecules independently, allowing similarity analysis via simple comparisons rather than a forward pass per pair. However, standard CLIP training was designed for symmetric tasks and does not account for the asymmetric and one-to-many nature of protein-molecule binding. In this paper, we introduce *BindScreen*, a sequence-based virtual screening method built on a dual-encoder contrastive architecture. BindScreen introduces a protein-centric batch construction strategy and an asymmetric multi-positive InfoNCE loss to cope with the protein-centric nature of virtual screening. We conducted a systematic evaluation of 8 protein language models and 3 molecular language model variants against BindScreen. The proposed protein-centric batch construction consistently outperforms standard CLIP training across all evaluated encoders while substantially improving computational efficiency, reducing training cost by up to 32 times. In addition, our experiments demonstrate that BindScreen requires 7 times fewer inference computations than pairwise virtual screening approaches. On the LIT-PCBA dataset, BindScreen outperforms all sequence-based baselines, achieving a relative improvement of up to 39% in EF at 0.5 over the best competing method, while remaining competitive with traditional docking approaches without requiring 3D structural information.

## System Requirements
- A computer with Ubuntu 16.04 (or later) or CentOS 8.1 (or later).
- CUDA-enabled GPU with at least 6 GB of memory.

## Installation Guide

### Install Anaconda
[Step by Step Guide to Install Anaconda](https://docs.anaconda.com/anaconda/install/)


### Fork the Repository
- Fork this repository to your own account.
- Clone your fork to your machine.

### Create a Conda Environment
```bash
cd <repository_directory>
conda env create --file environment.yml
```

### Activate the Environment
```bash
conda activate bindscreen
```

## Running the Experiments

1. Extract proteins and molecules embeddings:
```bash
cd src
python extract.py --modality protein
```

Arguments:
- **--modality**: Modality (protein or molecule)

2. Train and evaluate:
```bash
cd src
python train.py --mode embedding --dataset chembl
```

Arguments:
- **--mode**: Embedding mode (referred to as frozen in the paper - use **--mode embedding**) or Tokenized mode (referred to as finetuning in the paper - use **--mode tokenized**)
- **--dataset**: Dataset for training and evaluation on the test set (chembl or lit_pcba)

---

## Citation

The paper is under review. As soon as it is accepted, we will update this section.

## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author.

## Contact

For any additional questions or comments, contact Fahad Saeed (fsaeed@fiu.edu).

