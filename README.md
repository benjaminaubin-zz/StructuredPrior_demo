# The spiked matrix model with generative priors

We provide demonstration codes associated to the article arXiv: <http://arxiv.org/>

###
  * [Python code for AMP, LAMP, PCA](../Demo_AMP_SE_Spectral)
  * [Python code on MNIST/FashionMNIST Dataset](../Demo_MNIST)
  * [Mathematica notebook for random matrix theory computatuions](../Demo_RMT)



## Abstract
Using a low-dimensional parametrization of signals is a generic and powerful way to enhance performance in signal processing and statistical inference. A very popular and widely explored type of dimensionality reduction is sparsity; another
type is generative modelling of signal distributions. Generative models based on neural networks, such as GANs or variational auto-encoders,
are particularly performant and are gaining on applicability. In this paper we study spiked matrix models, where a low-rank matrix is observed through a noisy
channel. This problem with sparse structure of the spikes has attracted
broad attention in the past literature. Here, we replace the sparsity assumption by
generative modelling, and investigate the consequences on statistical and
algorithmic properties. We analyze the Bayes-optimal
performance under specific generative models for the spike. In contrast with
the sparsity assumption, we do not observe regions of parameters where
statistical performance is superior to the best known algorithmic
performance. We show that in the analyzed cases the approximate
message passing algorithm is able to reach optimal performance. We also design
enhanced spectral algorithms and analyze their performance and
thresholds using random matrix theory, showing their superiority to the
classical principal component analysis. We complement our theoretical
results by illustrating the performance of the spectral algorithms when the spikes come from real datasets.

*Benjamin Aubin, Bruno Loureiro, Antoine Maillard, Florent Krzakala, Lenka Zdeborova
