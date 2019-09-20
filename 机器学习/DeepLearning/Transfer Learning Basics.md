[paper pdf](https://arxiv.org/pdf/1407.4979.pdf)

#Transfer learning basics

## Three types of transfer learning

* **weight transfer**: a model trained on the source domain is used as an **initialization point** for a network to be trained on the target domain

* **deep metric learning**: the source domain is used to construct an embedding that captures class structure in both the source and target domains.

   for example, in person re-identification, use two neural networks to generate two feature vectors, and then use a network to learn a similarity metric (With designed loss function). ([Pedestrian Recognition with a Learned Metric [ACCV2010]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.389.5887&rep=rep1&type=pdf))

* **few shot learning**: learn new concepts from only a few samples.  

  In [Siamese Neural Networks for One-shot Image Recognition [ICML2015]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf),  author uses **Siamese Network** (A shared network) to get feature pair $(f(x_1), f(x_2))​$  from sample pair $(x_1, x_2)​$, then using a weighted L1 distance with sigmoid activation to compute similarity. The loss function is cross entropy with label $\mathop{y}\limits ^{\wedge} = \left\{ \begin{array}{}  1 \quad class(x_1) = class(x_2) \\0 \quad \text{otherwise} \end{array} \right.​$

  I have implemented the code for this paper, and have got desired accuracy. 

  *Note: The learning rate should be very low otherwise the model cannot converge and all predictions tend to be 0.5*

  

  

## Metric Learning

Metric learning is to find a good similarity(distance) measurement for data. And then the metric can used for clustering or classification. And some time we have too many classes(like person-reidentification), where   it is not possible to find a model directly output the sample's class, we can use the learned metric to determine whether two samples are belong to the same kind or not.

In paper [Distance Metric Learning for Large Margin Nearest Neighbor Classiﬁcation [NIPS2006] [JMLR2009]](http://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf), author described several classic distance metric learning methods, mostly Mahalanobis distances ($\mathrm{d}(\mathbf{x}, \mathbf{y}) = \mathbf{x} M \mathbf{y} \quad \text{with } M \in S^{++}​$) include:

* 