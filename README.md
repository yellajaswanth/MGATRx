# MGATRx: Discovering Drug Repositioning Candidates Using Multi-view Graph Attention

Authors: Jaswanth K. Yella, Anil G. Jegga

MGATRx predicts novel drug–disease associations by learning node embeddings on a
heterogeneous biological knowledge graph using multi-view graph attention networks.
It integrates six views of drug and disease annotations (targets, chemical fingerprints,
side effects, MeSH categories) and outputs ranked drug repositioning candidates.

## Abstract

```
In-silico drug repositioning or predicting new indications for approved or late-stage clinical trial drugs is a resourceful and time-efficient strategy in drug discovery. However, inferring novel candidate drugs for a disease is challenging, given the heterogeneity and sparseness of the underlying biological entities and their relationships (e.g., disease/drug annotations). By integrating drug-centric and disease-centric annotations as multiviews, we propose a multi-view graph attention network for indication discovery (MGATRx). Unlike most current similarity-based methods, we employ graph attention network on the heterogeneous drug and disease data to learn the representation of nodes and identify associations. MGATRx outperformed four other state-of-art methods used for computational drug repositioning. Further, several of our predicted novel indications are either currently investigated or are supported by literature evidence, demonstrating the overall translational utility of MGATRx.
```

## Architecture

![MGATRx architecture](https://i.ibb.co/kxw78yV/figure.png)

*Figure: MGATRx encodes a heterogeneous graph of drugs, diseases, targets, chemical
fingerprints, side effects, and MeSH categories using stacked HeteroGCN or HeteroGAT
layers. A per-task linear decoder reconstructs the drug–disease adjacency matrix from
the learned node embeddings.*

## Project Structure

```
MGATRx/
├── MGATRx.py              # Entry point: argument parsing, K-fold loop, result logging
├── source/
│   ├── models.py          # MGATRx, HeteroGCN, HeteroGAT model classes
│   ├── layers.py          # GraphConvolution, CosineGraphAttentionLayer, DictReLU/Dropout
│   ├── trainer.py         # build_model, calculate_loss, train_fold
│   ├── evaluate.py        # aggregate_fold_predictions, compute_and_log_metrics
│   ├── metrics.py         # AUC, AUPR, AP@k, F1 helper functions
│   ├── utils.py           # Data loading, adjacency normalization, sparse helpers
│   └── argparser.py       # CLI argument definitions
├── tests/                 # Unit tests (pytest)
│   ├── test_layers.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_utils.py
├── data/
│   └── DB-KEGG.zip        # Dataset archive (see Data Setup below)
└── requirements.txt
```

## Dependencies

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

> **Note:** `torch`, `torch-sparse`, and `torch-geometric` require matching CUDA versions.
> See the [PyTorch install guide](https://pytorch.org/get-started/locally/) and
> [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> for platform-specific instructions.

| Package | Min Version |
|---|---|
| torch | 1.9.0 |
| torch-sparse | 0.6.12 |
| torch-geometric | 2.0.0 |
| numpy | 1.21.0 |
| pandas | 1.3.0 |
| scikit-learn | 0.24.2 |
| networkx | 2.6.0 |
| tqdm | 4.62.0 |
| scipy | 1.7.0 |

## Data Setup

Unzip the dataset archive before running:

```bash
unzip data/DB-KEGG.zip -d data/
```

This creates `data/DB-KEGG/` with the following input files:

| File | Description | Shape |
|---|---|---|
| `drug-disease.txt` | Drug–disease association matrix (labels) | drugs × diseases |
| `drug-targets.txt` | Drug–target interaction matrix | drugs × targets |
| `drug-chemfp.txt` | Drug chemical fingerprint matrix | drugs × substructures |
| `drug-se.txt` | Drug side effect annotation matrix | drugs × side effects |
| `drug-meshcat.txt` | Drug MeSH category annotation matrix | drugs × MeSH categories |
| `disease-targets.txt` | Disease–target association matrix | diseases × targets |

All files are space-delimited binary matrices loadable via `numpy.loadtxt`.

## Usage

```bash
# Train with default settings (10-fold CV, GAT encoder, 2000 epochs)
python MGATRx.py

# Run only the first fold — useful for quick debugging
python MGATRx.py --fold-test --epochs 100

# Use GCN encoder with a smaller embedding size
python MGATRx.py --encoder GCN --embed-size 256

# Tune learning rate, dropout, and weight decay
python MGATRx.py --lr 0.005 --dropout 0.2 --weight-decay 1e-4

# Save per-fold model checkpoints to tmp/
python MGATRx.py --save-model
```

**All CLI options:**

| Argument | Default | Description |
|---|---|---|
| `--encoder` | `GAT` | Encoder backbone: `GCN` or `GAT` |
| `--decoder` | `linear` | Decoder type (only `linear` is active) |
| `--encoder-activation` | `selu` | Activation: `leaky`, `selu`, `relu`, `prelu`, `tanh`, `sigmoid`, `elu`, `none` |
| `--embed-size` | `512` | Node embedding dimensionality |
| `--num-layers` | `1` | Number of encoder layers |
| `--epochs` | `2000` | Maximum training epochs per fold |
| `--lr` | `0.01` | Adam learning rate |
| `--weight-decay` | `0` | Adam L2 weight decay |
| `--dropout` | `0.0` | Dropout rate in predictor head |
| `--kfolds` | `10` | Number of stratified K-folds |
| `--valid-size` | `0.15` | Fraction of train set used for validation |
| `--seed` | `1` | Random seed |
| `--save-model` | `False` | Save fold checkpoints to `tmp/` |
| `--fold-test` | `False` | Run only the first fold (debug mode) |

Outputs are written to `logs/`: a TSV metrics log and per-fold epoch-vs-AUPR traces.

## Running Tests

```bash
pytest tests/ -v
```

## Citation

To cite this [paper](https://doi.ieeecomputersociety.org/10.1109/TCBB.2021.3082466), please use this bibtex entry:

```BibTeX
@ARTICLE{9437764,
  author  = {Yella, Jaswanth K. and Jegga, Anil G.},
  journal = {IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  title   = {MGATRx: Discovering Drug Repositioning Candidates Using Multi-view Graph Attention},
  year    = {2021},
  volume  = {19},
  number  = {4},
  pages   = {2608--2618},
  doi     = {10.1109/TCBB.2021.3082466},
  issn    = {1557-9964},
  month   = {may}
}
```

## Acknowledgment

This work was supported, in part, by NIH NCATS grant.
