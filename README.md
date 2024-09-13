# MEB AutoGrad

Obviously there are many autodifferentation tools, especially in Python. This implementation, like those, mainly uses NumPy as it's only dependency. The main difference is the implementation, this is a tool to be used in my MPNN implementations for chemical systems.

## Some short notes

Inspiration taken from [here](https://github.com/smolorg/smolgrad/blob/master/smolgrad/core/engine.py) and [here](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py) but used for implementation in papers like [this](https://arxiv.org/abs/1706.08566) and [this](https://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf). The public repos are WIP. 

Uses simple topo sort, for applying chain rule.
