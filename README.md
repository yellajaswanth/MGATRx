# MGATRx: Discovering Drug Repositioning Candidates Using Multi-view Graph Attention

Authors: Jaswanth K. Yella, Anil G. Jegga

## Abstract
In-silico drug repositioning or predicting new indications for approved or late-stage clinical trial drugs is a resourceful and time-efficient strategy in drug discovery. However, inferring novel candidate drugs for a disease is challenging, given the heterogeneity and sparseness of the underlying biological entities and their relationships (e.g., disease/drug annotations). By integrating drug-centric and disease-centric annotations as multiviews, we propose a multi-view graph attention network for indication discovery (MGATRx). Unlike most current similarity-based methods, we employ graph attention network on the heterogeneous drug and disease data to learn the representation of nodes and identify associations. MGATRx outperformed four other state-of-art methods used for computational drug repositioning. Further, several of our predicted novel indications are either currently investigated or are supported by literature evidence, demonstrating the overall translational utility of MGATRx.

![image](https://i.ibb.co/kxw78yV/figure.png)

## Dependencies
* PyTorch
* PyTorch Sparse
* PyTorch Geometric
* Numpy
* Pandas
* Scikit-learn
* Networkx

## Citation

To cite this [paper](https://doi.ieeecomputersociety.org/10.1109/TCBB.2021.3082466), please use this bibtex entry:

```BibTeX
@ARTICLE {9437764,
author = {J. Yella and A. Jegga},
journal = {IEEE/ACM Transactions on Computational Biology and Bioinformatics},
title = {MGATRx: Discovering Drug Repositioning Candidates Using Multi-view Graph Attention},
year = {5555},
volume = {},
number = {01},
issn = {1557-9964},
pages = {1-1},
keywords = {drugs;diseases;annotations;graph neural networks;bioinformatics;mathematical model;feature extraction},
doi = {10.1109/TCBB.2021.3082466},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {may}
}
```

## Acknowledgment
This work was supported, in part, by NIH NCATS grant.

