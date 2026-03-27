# MGATRx: Discovering Drug Repositioning Candidates Using Multi-view Graph Attention

Authors: Jaswanth K. Yella, Anil G. Jegga

## Abstract
In-silico drug repositioning or predicting new indications for approved or late-stage clinical trial drugs is a resourceful and time-efficient strategy in drug discovery. However, inferring novel candidate drugs for a disease is challenging, given the heterogeneity and sparseness of the underlying biological entities and their relationships (e.g., disease/drug annotations). By integrating drug-centric and disease-centric annotations as multiviews, we propose a multi-view graph attention network for indication discovery (MGATRx). Unlike most current similarity-based methods, we employ graph attention network on the heterogeneous drug and disease data to learn the representation of nodes and identify associations. MGATRx outperformed four other state-of-art methods used for computational drug repositioning. Further, several of our predicted novel indications are either currently investigated or are supported by literature evidence, demonstrating the overall translational utility of MGATRx.

![image](https://i.ibb.co/kxw78yV/figure.png)

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

This creates `data/DB-KEGG/` with the required input files.

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

