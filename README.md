# ParamCorrpution
Reimplementation of AAAI21 Paper: [Exploring the Vulnerability of Deep Neural Networks:
A Study of Parameter Corruption](https://arxiv.org/pdf/2006.05620.pdf)

Download and unzip the MNIST dataset with:
> wget www.di.ens.fr/~lelarge/MNIST.tar.gz
> 
> tar -xzvf MNIST.tar.gz

## Random Corruption

Randomly corrupt the model params with noise $\epsilon * \mathcal{N}(0, I) $,
the results before and after attacking under different magnitudes are give below for a simple linear MNIST classifier:

| Model        | Eps=1e-3     | Eps=1e-2      | 1e-1        | Eps=1e0     | Eps=1e2       |
| ------------ | ------------ | ------------- | ----------- | ----------- | ------------- |
| Linear Model | 96.23 /96.26 | 96.23 / 96.26 | 96.23/83.71 | 96.23/11.26 | 96.23 / 10.64 |
|              |              |               |             |             |               |



