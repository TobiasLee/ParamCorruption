# Parameter Corrpution
Reimplementation of AAAI21 Paper: [Exploring the Vulnerability of Deep Neural Networks:
A Study of Parameter Corruption](https://arxiv.org/pdf/2006.05620.pdf)

Download and unzip the MNIST dataset with:
> wget www.di.ens.fr/~lelarge/MNIST.tar.gz
> 
> tar -xzvf MNIST.tar.gz

## Results

- Random Attack
Corrupt the model params with noise $\epsilon * \mathcal{N}(0, I) $ ;
  
- Grad-based: Corrupt the model params with the most indicative direction, see paper for details.

The results before and after attacking under different magnitudes are give below for a simple linear MNIST classifier:

| Attack / Eps      | 1e-3     | 1e-2      | 1e-1        | 1e0     | 1e2       |
| ------------ | ------------ | ------------- | ----------- | ----------- | ------------- |
| Random Attack | 96.23 / 96.26 | 96.23 / 96.26 | 96.23 / 83.71 | 96.23 / 11.26 | 96.23 / 10.64 |
| Grad-based Attack, lr=1e-3 |   96.23 / 96.23     |  96.23 / 96.14  | 96.23 / 94.57 | 96.23 / 66.72 |  96.23 / 66.72  |
| Grad-based Attack, lr=1 |   96.23 / 96.25     |  96.23 / 96.25  | 96.23 /  95.89| 96.23 / 31.52  |  96.23 / 8.92  |



