# RACS: Risk-Aware Cold-Start Recommendation with LLM Contrastive Scoring
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

RACS is a powerful recommender system framework designed to tackle **extreme cold-start scenarios** (where user history is very limited) using Large Language Models (LLMs). It balances user preference alignment with safety and uncertainty-driven exploration.

## 🚀 Key Features
* **ACS (Alignment Contrastive Scoring):** Uses LLMs (like Llama 3.1) to contrastively rank items by simulating user engagement.
* **SND (Semantic Neighborhood Disagreement):** A Bayesian approach to measure semantic uncertainty, acting as an epistemic exploration bonus.
* **Risk-Awareness:** Implements a safety constraint that penalizes items with high risk scores (e.g., polarizing or unsafe content).
* **Plug-and-Play Baselines:** Includes high-performance baselines like **EASE**, **SASRec**, and **ItemKNN** for rigorous benchmarking.
* **Multi-Seed Evaluation:** Comprehensive experimental harness reporting mean ± std, bootstrap CIs, and paired Wilcoxon p-values.

## 🛠️ Installation
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/racs.git](https://github.com/YOUR_USERNAME/racs.git)
cd racs

# Install dependencies
pip install torch transformers faiss-gpu numpy pandas matplotlib scikit-learn tqdm
```
*Note: For the fastest performance, we recommend using a GPU and installing flash-attn.*

## 📂 Project Structure
```text
ksai/
├── online.py           # Core RACS algorithm logic (ACS, SND, Risk)
├── run_online.py       # Main entry point for evaluation and experiments
├── offline_emb.py      # Offline phase: item embeddings & catalog building
├── hf_model.py         # HuggingFace LLM/Embedder backend
├── calibrate_risk.py   # Risk score quantile-rescaling tool
├── data_loaders.py     # Multi-dataset support (MovieLens, Semantic Scholar)
├── replay.py           # Sequential and static evaluation harness
├── sasrec.py           # SASRec sequential baseline
├── itemknn.py          # ItemKNN baseline
└── final_report.tex    # LaTeX source for the research report
```

## 📖 Usage Guide

### 1. Offline Phase (Embedding & Risk Scoring)
First, generate item embeddings and calculate initial risk scores for your dataset.

```bash
python -m ksai.offline_emb --cache-dir /PATH/TO/CACHE --model BAAI/bge-large-en-v1.5
```

### 2. Risk Calibration
Rescale the risk scores to a uniform [0, 1] distribution to ensure the risk penalty is numerically stable.

```bash
python -m ksai.calibrate_risk --cache-dir /PATH/TO/CACHE
```


## 📊 Experimental Suite
The system includes 6 built-in experiments:

1. **Main Cold-Start Table:** Comparison of RACS against all baselines.
2. **Learning Curve:** Performance as more user history is revealed.
3. **Component Isolation:** Full ablation study of ACS, SND, and Risk terms.
4. **Risk Penalty Validation:** Verifying the "Skip Rate" reduction during exploration.
5. **Cross-Domain Consistency:** Testing robustness across different datasets.
6. **Risk-Recall Trade-off:** Sweeping λ to find the optimal safety-accuracy balance.

## 📝 Citation
If you use this work in your research, please cite:

```bibtex
@article{patil2026racs,
  title={RACS: Risk-Aware Generative Alignment for Cold-Start Recommendation},
  author={Patil, Areen and Parsania, Ramya and Velidanda, Krishna Sai},
  year={2026}
}
```

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
