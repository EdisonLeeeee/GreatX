# GraphWar: Arms Race in *Graph Adversarial Learning*

NOTE: GraphWar is still in the early stages and the API will likely continue to change.


# 🚀 Installation
Please make sure you have installed [PyTorch](https://pytorch.org/) and [Deep Graph Library(DGL)](https://www.dgl.ai/pages/start.html).
```bash
# Comming soon
pip install -U graphwar
```
or
```bash
# Recommended
git clone https://github.com/EdisonLeeeee/GraphWar.git && cd GraphWar
pip install -e . --verbose
```
where `-e` means "editable" mode so you don't have to reinstall every time you make changes.

# Get Started
## A simple targeted attack
```python
from graphwar.attack.targeted import RandomAttack
attacker = RandomAttack(g)
attacker.attack()
attacked_g = attacker.g()
edge_flips = attacker.edge_flips()

```

## A simple untargeted attack
```python
from graphwar.attack.untargeted import RandomAttack
attacker = RandomAttack(g)
attacker.attack()
attacked_g = attacker.g()
edge_flips = attacker.edge_flips()

```


# Implementations

In detail, the following methods are currently implemented:

## Attack

### Targeted Attack

| Methods | Venue |
| ---------------- | ------------------------------------------------------------ |
|**RandomAttack** | A simple random method that choose edges to flip randomly. |
|**DICEAttack** | *Marcin Waniek et al*, [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), *Nature Human Behavior'16* |
|**Nettack** | *Daniel Zügner et al.*, [📝Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/abs/1805.07984), *KDD'18* |
|**FGAttack** | *Jinyin Chen et al.* [📝Fast Gradient Attack on Network Embedding](https://arxiv.org/abs/1809.02797), arXiv'18<br>*Jinyin Chen et al.* [📝Link Prediction Adversarial Attack Via Iterative Gradient Attack](https://ieeexplore.ieee.org/abstract/document/9141291), IEEE Trans' 20 </br> <br> *Hanjun Dai et al.* [📝Adversarial Attack on Graph Structured Data](https://arxiv.org/abs/1806.02371), ICML'18 </br> |
|**GFAttack** | *Heng Chang et al*.  [📝A Restricted Black - box Adversarial Framework Towards Attacking Graph Embedding Models](https://arxiv.org/abs/1908.01297), *AAAI'20* |
|**IGAttack** | *Huijun Wu et al.* [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), *IJCAI'19* |
|**SGAttack** | *Jintang Li et al.* [📝 Adversarial Attack on Large Scale Graph](https://arxiv.org/abs/2009.03488), *TKDE'21* |

### Untargeted Attack

| Methods | Venue |
| ------------------------- | ------------------------------------------------------------ |
|**RandomAttack** | A simple random method that choose edges to flip randomly |
|**DICEAttack** | *Marcin Waniek et al*, [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), *Nature Human Behavior'16* |
|**FGAttack** | *Jinyin Chen et al.* [📝Fast Gradient Attack on Network Embedding](https://arxiv.org/abs/1809.02797), *arXiv'18*<br/>*Jinyin Chen et al.* [📝Link Prediction Adversarial Attack Via Iterative Gradient Attack](https://ieeexplore.ieee.org/abstract/document/9141291), *IEEE Trans'20* </br> <br/>*Hanjun Dai et al.* [📝Adversarial Attack on Graph Structured Data](https://arxiv.org/abs/1806.02371), *ICML'18* </br> |
|**Metattack** | *Daniel Zügner et al.* [📝Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412), *ICLR'19* |
|**PGD**, **MinmaxAttack** | *Kaidi Xu et al.* [📝Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), *IJCAI'19* |

## Defense

### Model-Level

| Methods | Venue |
| ------------- | ------------------------------------------------------------ |
|**MedianGCN** | *Liang Chen et al.* [📝Understanding Structural Vulnerability in Graph Convolutional Networks](https://www.ijcai.org/proceedings/2021/310), *IJCAI'21* |
|**RobustGCN** | *Dingyuan Zhu et al.*  [📝Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf), *KDD'19* |

### Data-Level

| Methods | Venue |
| ----------------------- | ------------------------------------------------------------ |
|**JaccardPurification** | *Huijun Wu et al.* [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), *IJCAI'19* |


More details of literatures and the official codes can be found in [Awesome Graph Adversarial Learning](https://github.com/gitgiter/Graph-Adversarial-Learning).
