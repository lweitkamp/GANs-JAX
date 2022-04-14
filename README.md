# Generative Adversarial Networks in JAX

This repository holds several notebooks that implement GANs in JAX using the Flax Linen package. All models are trained on Colab using the MNIST dataset on TPUs, with parallelization enabled by default.

## Todo's
I still want to add additional models to this repository. I'm working on the following additions:

- [ ] StyleGAN
- [ ] CycleGAN

## Deep Convolutional GAN
The <a href="https://arxiv.org/abs/1406.2661">original GAN</a> with architecture and other tips from the <a href="https://arxiv.org/abs/1511.06434">GANs for representation learning</a> paper.


## Wasserstein GAN with Penality
Training GANs is a notoriously difficult process.
Even by carefully selecting the model architecture, training can still suffer due to mode collapse.
The authors of the <a href="https://arxiv.org/abs/1701.07875">Wasserstein GAN</a> paper argue the biggest problem is the way that the vanilla GAN learns a distribution; by switching to minimizing the earth mover distance we can alleviate this problem.



## Conditional GAN
<a href="https://arxiv.org/abs/1411.1784">This is the logical next step</a> after the vanilla GAN.
If we *do* have labels, we should utilize them somehow.
The Conditional GAN, as the name implies, conditions the output of the generator on the labels in addition to the noise.
The discriminator in turn receives both the generated/real images and the label for classification.


## InfoGAN
My personal favorite is the <a href="https://arxiv.org/pdf/1606.03657">information-maximizing GAN</a>.
As the authors mention, because the info loss converges faster than the GAN loss, this addition basically comes for free.
The result is a somewhat disentangled latent space where digits are easily separable.
A great reference and interpretation of both the InfoGAN objective and the vanilla objective can be found <a href="https://www.inference.vc/infogan-variational-bound-on-mutual-information-twice/">here</a> in Ferenc Huszár's blog.
