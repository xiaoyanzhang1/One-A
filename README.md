# One Adapter for All: Towards Unified Representation in Step-Imbalanced Class-Incremental Learning

This repository provides the official PyTorch implementation of **One-A** for step-imbalanced class-incremental learning (SI-CIL), including training scripts and experiment configurations for:
- **CIFAR100**
- **CUB200**
- **ImageNet-A**
- **ImageNet-R**

> **Note:** This repository **does not** include pretrained model weights. Backbones are loaded via `timm` (when applicable).




## 💻 Environment

```bash
conda env create -f environment.yaml
conda activate onea
```


## ▶️ Running scripts
After adjusting the config file (if necessary), you can launch training with:
```bash
python main.py --config=./exps/[dataset_config].json
```

Example: run One-A on CIFAR-100 with the provided step-imbalanced setting:
```bash
python main.py --config=./exps/cifar.json
```

## 🤝 Acknowledgement
We would like to thank the repository [PILOT: A Pre-Trained Model-Based Continual Learning Toolbox](https://github.com/sun-hailong/LAMDA-PILOT) for providing important components and the overall training pipeline on which One-A is built.

## 🌟 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@misc{zhang2026adapterallunifiedrepresentation,
      title={One Adapter for All: Towards Unified Representation in Step-Imbalanced Class-Incremental Learning}, 
      author={Xiaoyan Zhang and Jiangpeng He},
      year={2026},
      eprint={2603.10237},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.10237}, 
}
```
