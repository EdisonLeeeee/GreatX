# ⚔🛡 GraphWar: Arms Race in *Graph Adversarial Learning*

The robustness of graph neural networks (GNNs) against adversarial attacks has gained increasing attention in last few years. While
there are numerous (heuristic) approaches aimed at robustifying GNNs, there is always a newly devised stronger attack attempts to break them, leading to an arms race between attackers and defenders. To this end, GraphWar aims to provide easy implementations with unified interfaces to facilitate the research in Graph Adversarial Learning.

---

NOTE: GraphWar is still in the early stages and the API will likely continue to change. 

If you are interested in this project, don't hesitate to contact me or make a PR directly.


# 🚀 Installation

Please make sure you have installed [PyTorch](https://pytorch.org) and [Deep Graph Library (DGL)](https://www.dgl.ai/pages/start.html).

```bash
# Comming soon
pip install -U graphwar
```

or

```bash
# Recommended now
git clone https://github.com/EdisonLeeeee/GraphWar.git && cd GraphWar
pip install -e . --verbose
```

where `-e` means "editable" mode so you don't have to reinstall every time you make changes.

# ⚡ Get Started

Assume that you have a `dgl.DGLgraph` instance `g` that describes your dataset.
NOTE: Please make sure that `g` DO NOT contain selfloops, i.e., run `g = g.remove_self_loop()`.

## A simple manipulation targeted attack

```python
from graphwar.attack.targeted import RandomAttack
attacker = RandomAttack(g)
attacker.attack(1, num_budgets=3) # attacking target node `1` with `3` edges 
attacked_g = attacker.g()
edge_flips = attacker.edge_flips()

```

## A simple manipulation  untargeted (non-targeted) attack

```python
from graphwar.attack.untargeted import RandomAttack
attacker = RandomAttack(g)
attacker.attack(num_budgets=0.05) # attacking the graph with 5% edges perturbations
attacked_g = attacker.g()
edge_flips = attacker.edge_flips()

```


# 👀 Implementations

In detail, the following methods are currently implemented:

## Attack

### Manipulation Attack

#### Targeted Attack

| Methods             | Venue                                                        |
| ------------------- | ------------------------------------------------------------ |
| **RandomAttack**    | A simple random method that chooses edges to flip randomly.  |
| **DICEAttack**      | *Waniek et al.* [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), *Nature Human Behavior'16* |
| **Nettack**         | *Zügner et al.* [📝Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/abs/1805.07984), *KDD'18* |
| **FGAttack (FGSM)** | *Goodfellow et al.* [📝Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), *ICLR'15*<br>*Chen et al.* [📝Fast Gradient Attack on Network Embedding](https://arxiv.org/abs/1809.02797), *arXiv'18*<br>*Chen et al.* [📝Link Prediction Adversarial Attack Via Iterative Gradient Attack](https://ieeexplore.ieee.org/abstract/document/9141291), *IEEE Trans'20* <br> *Dai et al.* [📝Adversarial Attack on Graph Structured Data](https://arxiv.org/abs/1806.02371), ICML'18 </br> |
| **GFAttack**        | *Chang et al*.  [📝A Restricted Black - box Adversarial Framework Towards Attacking Graph Embedding Models](https://arxiv.org/abs/1908.01297), *AAAI'20* |
| **IGAttack**        | *Wu et al.* [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), *IJCAI'19* |
| **SGAttack**        | *Li et al.* [📝 Adversarial Attack on Large Scale Graph](https://arxiv.org/abs/2009.03488), *TKDE'21* |

#### Untargeted Attack

| Methods                   | Venue                                                        |
| ------------------------- | ------------------------------------------------------------ |
| **RandomAttack**          | A simple random method that chooses edges to flip randomly   |
| **DICEAttack**            | *Waniek et al.* [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), *Nature Human Behavior'16* |
| **FGAttack (FGSM)**       | *Goodfellow et al.* [📝Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), *ICLR'15*<br>*Chen et al.* [📝Fast Gradient Attack on Network Embedding](https://arxiv.org/abs/1809.02797), *arXiv'18*<br>*Chen et al.* [📝Link Prediction Adversarial Attack Via Iterative Gradient Attack](https://ieeexplore.ieee.org/abstract/document/9141291), *IEEE Trans'20* <br> *Dai et al.* [📝Adversarial Attack on Graph Structured Data](https://arxiv.org/abs/1806.02371), *ICML'18* </br> |
| **Metattack**             | *Zügner et al.* [📝Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412), *ICLR'19* |
| **PGD**, **MinmaxAttack** | *Xu et al.* [📝Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), *IJCAI'19* |

### Injection Attack

### Universal Attack

### Backdoor Attack

| Methods                         | Venue                                                        |
| ------------------------------- | ------------------------------------------------------------ |
| **LGCBackdoor**, **FGBackdoor** | *Chen et al.* [📝Neighboring Backdoor Attacks on Graph Convolutional Network](https://arxiv.org/abs/2201.06202), *arXiv'22* |



## Defense

### Standard GNNs (without defense)

| Methods   | Venue                                                        |
| --------- | ------------------------------------------------------------ |
| **GCN**   | *Kipf et al.* [📝Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), *ICLR'17* |
| **SGC**   | *Wu et al.*  [📝Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153), *ICLR'19* |
| **GAT**   | *Veličković et al.*  [📝Graph Attention Networks](https://arxiv.org/abs/1710.10903), *ICLR'18* |
| **DAGNN** | *Liu et al.*  [📝Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296), *KDD'20* |
| **APPNP** | *Klicpera et al.*  [📝Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997), *ICLR'19* |
| **JKNet** | *Xu et al.*  [📝Representation Learning on Graphs with Jumping Knowledge Networks](hhttps://arxiv.org/abs/1806.03536), *ICML'18* |

### Model-Level

| Methods         | Venue                                                        |
| --------------- | ------------------------------------------------------------ |
| **MedianGCN**   | *Chen et al.* [📝Understanding Structural Vulnerability in Graph Convolutional Networks](https://www.ijcai.org/proceedings/2021/310), *IJCAI'21* |
| **RobustGCN**   | *Zhu et al.*  [📝Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf), *KDD'19* |
| **ReliableGNN** | *Geisler et al.* [📝Reliable Graph Neural Networks via Robust Aggregation](https://arxiv.org/abs/2010.15651), *NeurIPS'20*<br>*Geisler et al.* [📝Robustness of Graph Neural Networks at Scale](https://arxiv.org/abs/2110.14038), *NeurIPS'21* |
| **ElasticGNN**  | *Liu et al.* [📝Elastic Graph Neural Networks](https://arxiv.org/abs/2107.06996), *ICML'21* |
| **AirGNN**      | *Liu et al.* [📝Graph Neural Networks with Adaptive Residual](https://openreview.net/forum?id=hfkER_KJiNw), *NeurIPS'21* |
| **SimPGCN**      | *Jin et al.* [📝Node Similarity Preserving Graph Convolutional Networks](https://arxiv.org/abs/2011.09643), *WSDM'21* |

### Data-Level

| Methods                 | Venue                                                        |
| ----------------------- | ------------------------------------------------------------ |
| **JaccardPurification** | *Wu et al.* [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), *IJCAI'19* |
| **SVDPurification**     | *Entezari et al.* [📝All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://arxiv.org/abs/1903.01610), *WSDM'20* |


More details of literatures and the official codes can be found at [Awesome Graph Adversarial Learning](https://github.com/gitgiter/Graph-Adversarial-Learning).
