# A Comparative Analysis of Traditional Machine Learning, Neural Networks, and Transformers for Multi-Class Text Classification

A reproducible empirical study comparing three families of NLP models — classical machine learning, recurrent neural networks, and pre-trained transformers — on a 10-class Question/Answer text classification task. The project benchmarks 25 experiments end-to-end across model families, input representations, and optimisation strategies, and reports performance, training cost, and architectural compatibility findings.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Hardware and Environment](#hardware-and-environment)
- [Authors](#authors)

## Overview

| Property | Value |
| --- | --- |
| Task | Multi-class text classification (10 classes) |
| Model families compared | Traditional ML, Neural Networks, Transformers |
| Total experiments | 25 |
| Best overall F1-macro | 0.742 (BERT-base-uncased) |
| Best non-transformer F1-macro | 0.694 (Logistic Regression + TF-IDF) |
| Implementation | Single Jupyter notebook, PyTorch backend |

The study evaluates how the choice of input representation (Bag-of-Words, TF-IDF, GloVe, Word2Vec, subword tokenisation) interacts with model architecture, and quantifies the performance-versus-cost trade-off when scaling from sparse linear models up to fine-tuned transformers.

## Dataset

**Question Answer Classification Dataset** — a 10-class corpus of question/answer text pairs.

| Statistic | Value |
| --- | --- |
| Training samples (source) | 279,999 |
| Test samples (source) | 59,999 |
| Total samples | 339,998 |
| Classes | 10 |
| Class imbalance ratio | 1.01 (effectively balanced) |
| Class entropy | 3.32 |
| Mean text length | 566 characters |
| Mean word count | 97 words |
| Development sample ratio | 2% stratified (scalable to full dataset) |

**Classes:** Business & Finance, Computers & Internet, Education & Reference, Entertainment & Music, Family & Relationships, Health, Politics & Government, Science & Mathematics, Society & Culture, Sports.

## Methodology

The notebook is organised as a sequential pipeline covering data loading, EDA, preprocessing, representation learning, and model training across three benchmark suites.

### Preprocessing pipeline

Text normalisation, tokenisation (NLTK), stopword removal, and lemmatisation. Word representations include sparse counts (BoW), TF-IDF, and dense pre-trained embeddings (GloVe, Word2Vec). Subword tokenisation is delegated to the respective transformer tokenisers.

### Traditional ML suite

| Setting | Detail |
| --- | --- |
| Models | Random Forest, Logistic Regression, Multinomial Naive Bayes |
| Representations | BoW, TF-IDF |
| Hyperparameter search | Optuna TPE sampler |
| Validation | 80/20 train-validation split, 3-fold CV |
| Metrics | Accuracy, F1-macro, F1-weighted, Precision, Recall |

DNNs with sparse BoW/TF-IDF inputs were intentionally excluded due to a documented architectural incompatibility (estimated dense-layer memory requirement ~293 GB versus 16 GB available); DNNs are instead evaluated with dense embeddings in the neural network suite.

### Neural network suite

| Setting | Detail |
| --- | --- |
| Architectures | DNN, SimpleRNN, GRU, LSTM, Bi-SimpleRNN, Bi-GRU, Bi-LSTM |
| Embeddings | GloVe, Word2Vec |
| Framework | PyTorch with CUDA |
| Optimiser | AdamW with OneCycleLR scheduler |
| Mixed precision | FP16 |
| Epochs | 15 |
| Batch size | 64 |
| Sequence length | 128 |

### Transformer suite

| Setting | Detail |
| --- | --- |
| Models | BERT-base-uncased, DistilBERT-base-uncased, RoBERTa-base |
| Fine-tuning | Full model fine-tuning with classification head |
| Optimiser | AdamW with linear warmup and decay |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size | 4 |
| Max sequence length | 128 |
| Warmup ratio | 0.1 |

## Results

### Best model per family (test set)

| Family | Best model | Representation | Accuracy | F1-macro | Parameters | Training time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Traditional ML | Logistic Regression | TF-IDF | 0.696 | 0.694 | — | 11.2 min |
| Neural Network | Bidirectional LSTM | GloVe | 0.689 | 0.684 | 633,866 | 3.3 min |
| Transformer | BERT-base-uncased | Subword | 0.746 | **0.742** | 109.5 M | 116.8 min |

### Traditional ML — complete results

| Model | Representation | Accuracy | F1-macro | Training time |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression | TF-IDF | 0.696 | 0.694 | 11.2 min |
| Naive Bayes | TF-IDF | 0.684 | 0.680 | 0.03 min |
| Logistic Regression | BoW | 0.686 | 0.684 | 131.5 min |
| Naive Bayes | BoW | 0.676 | 0.674 | 0.04 min |
| Random Forest | BoW | 0.617 | 0.608 | 169.5 min |
| Random Forest | TF-IDF | 0.616 | 0.608 | 186.9 min |

### Neural networks — complete results

| Architecture | Embedding | Accuracy | F1-macro | Parameters |
| --- | --- | ---: | ---: | ---: |
| Bidirectional LSTM | GloVe | 0.689 | 0.684 | 633,866 |
| DNN | GloVe | 0.689 | 0.683 | 187,146 |
| LSTM | GloVe | 0.687 | 0.682 | 896,010 |
| Bidirectional GRU | GloVe | 0.687 | 0.682 | 476,170 |
| GRU | GloVe | 0.687 | 0.681 | 672,778 |
| Bidirectional SimpleRNN | GloVe | 0.670 | 0.664 | 160,778 |
| SimpleRNN | GloVe | 0.669 | 0.663 | 226,314 |
| LSTM | Word2Vec | 0.615 | 0.611 | 896,010 |
| Bidirectional LSTM | Word2Vec | 0.615 | 0.610 | 633,866 |
| DNN | Word2Vec | 0.614 | 0.610 | 187,146 |
| GRU | Word2Vec | 0.614 | 0.609 | 672,778 |
| Bidirectional GRU | Word2Vec | 0.613 | 0.608 | 476,170 |
| Bidirectional SimpleRNN | Word2Vec | 0.590 | 0.583 | 160,778 |
| SimpleRNN | Word2Vec | 0.588 | 0.582 | 226,314 |

### Transformers — complete results

| Model | Accuracy | F1-macro | Parameters | Training time |
| --- | ---: | ---: | ---: | ---: |
| BERT-base-uncased | 0.746 | 0.742 | 109.5 M | 116.8 min |
| RoBERTa-base | 0.741 | 0.738 | 124.7 M | 124.3 min |
| DistilBERT-base-uncased | 0.740 | 0.736 | 67.0 M | 41.7 min |

Raw per-experiment metrics — including confusion matrices, classification reports, hyperparameters, and GPU memory traces — are persisted to [comprehensive_experimental_data_20250908_173940.json](comprehensive_experimental_data_20250908_173940.json). Summary tables for each suite are exported as `traditional_ml_results_table.xls`, `neural_network_results_table.xls`, and `transformer_results_table.xls`.

## Key Findings

1. **Transformers lead overall.** BERT-base improves F1-macro by 4.8 absolute points over the strongest traditional ML baseline and 5.8 points over the strongest RNN, at roughly 10× the training cost of the best neural network and 170× the parameter count of the best traditional ML model.
2. **Well-tuned classical ML remains competitive.** Logistic Regression with TF-IDF outperforms every recurrent architecture in this study while training in ~11 minutes on CPU.
3. **Representation choice dominates within the neural family.** Every architecture is ~7 F1-macro points higher with GloVe than with Word2Vec, a larger gap than the differences between RNN variants.
4. **DistilBERT is the efficiency sweet spot among transformers.** It loses only 0.6 F1-macro versus BERT-base while training in roughly a third of the time and using ~61% of the parameters.
5. **Architecture must match representation.** Dense neural networks fed sparse BoW/TF-IDF inputs were architecturally infeasible (~293 GB memory requirement). This motivates the use of dense embeddings for any neural pipeline.
6. **Bidirectionality helps less than expected.** Bi-LSTM/Bi-GRU/Bi-RNN gain less than 0.5 F1-macro points over their unidirectional counterparts at comparable parameter counts.

## Repository Structure

```
.
├── GroupNo08_24341256_22101497_21301538.ipynb   # Full experimental pipeline
├── GroupNo08_24341256_22101497_21301538.pdf     # IEEE-format paper write-up
├── comprehensive_experimental_data_20250908_173940.json  # Raw experimental data
├── traditional_ml_results_table.xls             # Traditional ML summary
├── neural_network_results_table.xls             # Neural network summary
├── transformer_results_table.xls                # Transformer summary
└── README.md
```

The notebook is self-contained and is organised into the following sections: environment setup, adaptive imports, data loading and EDA, text-based statistical analysis, preprocessing pipeline, word representation, traditional ML pipeline, neural network pipeline, transformer pipeline, results aggregation, and comparative analysis.

## Reproducing the Experiments

### Prerequisites

- Python 3.11 or newer (the original run used 3.13.1)
- CUDA-capable GPU recommended for the neural network and transformer suites
- Approximately 16 GB system RAM and 12 GB GPU VRAM for the full pipeline
- The Question Answer Classification Dataset CSV files placed under `C:/datasets/` (or update `data_path` in the configuration cell of the notebook)

### Setup

```bash
git clone https://github.com/aksaN000/A-Comparative-Analysis-of-Traditional-Machine-Learning.git
cd A-Comparative-Analysis-of-Traditional-Machine-Learning
python -m venv .venv
.venv\Scripts\activate              # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy pandas torch transformers scikit-learn scipy \
            matplotlib seaborn plotly nltk tqdm psutil optuna \
            gensim wordcloud textblob datasets ipywidgets
```

The first notebook cell performs an adaptive installation pass and tolerates missing optional packages (Gensim, WordCloud, TextBlob, Plotly) by falling back to alternative implementations.

### Running

Open the notebook in Jupyter or VS Code and execute cells sequentially:

```bash
jupyter notebook GroupNo08_24341256_22101497_21301538.ipynb
```

Each suite (traditional ML, neural networks, transformers) is independent. The configuration cell exposes `sample_ratio`, `use_full_dataset`, `num_epochs`, and other parameters for trading off run time against fidelity.

## Hardware and Environment

The reported results were produced on the following configuration:

| Component | Specification |
| --- | --- |
| CPU | AMD Ryzen 7 7700 (8 cores, 16 threads) |
| RAM | 16 GB |
| GPU | NVIDIA RTX 3060 12 GB |
| OS | Windows 11 |
| Python | 3.13.1 |
| PyTorch | 2.7.1 + CUDA 11.8 |
| pandas | 2.3.2 |
| numpy | 2.3.2 |

Optimisations applied during training include sparse matrix operations for classical pipelines, FP16 mixed precision and gradient accumulation for the neural and transformer suites, dynamic batch sizing based on available GPU memory, and aggressive memory cleanup between experiments.

## Authors

GroupNo08 — Course project.

| Student ID |
| --- |
| 24341256 |
| 22101497 |
| 21301538 |

## License

This repository is released for academic and educational purposes. Please cite the accompanying paper (see the included PDF) if you use any of the experimental data or methodology.
