# Adaptive Reasoning Activation Steering

All identified RCNs are stored in the `RCNs_Identification/analysis_results/` directory.

## Table of Contents

| # | Section |
|---|---------|
| 1 | [Setup](#setup) |
| 2 | [Quickstart](#quickstart) |
| 3 | [Overview](#overview) |
| &nbsp;&nbsp;3.1 | &nbsp;&nbsp;[Step 1 — RCNs Identification](#step-1-rcns-identification) |
| &nbsp;&nbsp;3.2 | &nbsp;&nbsp;[Step 2 — Training the Adaptive Classifier](#step-2-training-the-adaptive-classifier) |
| &nbsp;&nbsp;3.3 | &nbsp;&nbsp;[Step 3 — Evaluation](#step-3-evaluation) |

---

## Setup

```bash
conda create --name AdaRAS python=3.12
conda activate AdaRAS
pip install -r requirements.txt
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"
```

> **For users in China**, use the Tsinghua mirror to speed up installation:
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## Quickstart

We provide ready-to-run configurations for AIME-24 and AIME-25. Simply execute the following command and find the detailed results in the `results/` directory:

```bash
python run_experiment.py --config configs/AIME_XX.yaml
```

Replace `XX` with one of: `24_control`, `24_steering`, `25_control`, `25_steering`.

> **Note:** Due to hardware differences, we recommend using an **RTX 3090** to reproduce our results.

---

## Overview

### Step 1 — RCNs Identification

> Example using the AIME dataset with Qwen3-1.7B. Adjust `--dataset` and `--MODEL` for other configurations.

```bash
cd RCNs_Identification

# 1. Cache activations — extract MLP post-activations from the model
python cache_activations.py --dataset AIME --MODEL Qwen3-1.7B

# 2. Analyze neurons — compute steering weights and export a ranked CSV report
python analyze_and_refine_neurons_forsteering.py --dataset AIME --MODEL Qwen3-1.7B
```

---

### Step 2 — Training the Adaptive Classifier

> Example using the AIME dataset. Adjust `--dataset` and `--MODEL` for other configurations.

```bash
cd adaptive_classifier

# 1. Select features — rank neurons by F-score and save the top-N indices
python select_features.py --dataset AIME --MODEL Qwen3-1.7B --n_features 4096

# 2. Train classifier — train an attention-based classifier on the selected features
python train_attention_classifier.py --dataset AIME --MODEL Qwen3-1.7B --n_features 4096 --epochs 100

# 3. Evaluate classifier — evaluate on the validation set and save the optimal threshold
python evaluate_attention_classifier.py --dataset AIME --MODEL Qwen3-1.7B --n_features 4096
```

---

### Step 3 — Evaluation

> Run all commands from the project root directory (`AdaRAS/`).

```bash
# 1. Configure experiment — edit the relevant config file under configs/
#    Key fields: model_name, sample_path, intervention.enabled, intervention.n_features

# 2. Run experiment
python run_experiment.py --config configs/AIME_sample.yaml
```

Example config files for both math and code tasks are provided in `configs/`.

---

## Citation

```bibtex
@article{dong2026identifying,
  title={Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering},
  author={Dong, Fangan and Yan, Zuming and Ge, Xuri and Xu, Zhiwei and Zhang, Mengqi and Chen, Xuanang and He, Ben and Xin, Xin and Chen, Zhumin and Zhou, Ying},
  journal={arXiv preprint arXiv:2601.19847},
  year={2026}
}
```