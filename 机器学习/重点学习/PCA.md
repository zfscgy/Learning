主成分分析PCA全称 Principal Component Analysis，是一种常用的降维方法。网上很多对此的介绍并不是十分详细，本文打算对此进行完整地推导，并且用初等的方法证明PCA能够最小化其目标函数，有助于大家的理解。

## 引言

对于一列数据 $\{x_1, x_2, ...., x_m\}​$ ，其中每个 $x_i​$ 是一个 $N​$ 维的向量，则为了保存所有的数据信息，我们需要用 $ m \times n​$ 个实数。如何压缩这些信息？一个很好的方法就是降维，即把每个 $x_i​$ 通过一定的变换映射到一个 $K​$ 维的向量 $ y_i​$ 。

比如对于数据 $\{(0, 0), (0.5, 0.5), (1,1),(1.2,1.2), (3.5,3.5)\}​$，很容易看出来，我们可以把这个数据转化为一维的 $\{(0), (0.5), (1), (1.2), (3.5)\}​$, 然后再按照 $\mathbf x_i = \begin{bmatrix}1 \\ 1\end{bmatrix} \mathbf y_i​$ 来无损地恢复。

当然在一般情况下，我们不可能无损地保留所有数据，比如数据可能是 $\{(0, 0.2), (0.3, 0.5), (1,1.3),(1.1,1.2), (3.5,3.6)\}​$ ， 那么这种情况，如何进行压缩呢？

## 目标函数

我们假设压缩后，$x_i​$ 从$N​$ 维被压缩到了 $K​$ 维的 $y_i​$（此时假设 $x_i​$ 已经经过标准化处理，在各个维度的均值是0），然后再解压缩的时候通过一个 $N\times K​$ 的矩阵 $L​$ 来乘以 $y_i​$ 得到 $x_i​$ ，于是我们设置损失函数为 

$ \large Loss =\sum\limits_i (\mathbf x_i - L\mathbf y_i)^2=\sum\limits_i (\mathbf x_i - L\mathbf y_i)^T(\mathbf x_i - L\mathbf y_i)​$

至于为什么要用二次损失函数，应该可以从概率分布的角度进行理解，在此不进行赘述。

下面来考虑 $L​$ 的每一个行向量 $\mathbf l_1, \mathbf l_2, ..., \mathbf l_k​$ 。很容易可以看出，如果这些行向量不是单位正交向量，我们总是可以通过对$\mathbf y_i​$的各个分量 $y_{i1}, ..., y_{ik}​$ 进行变换让变换后的 $ L'\mathbf y' = L\mathbf y​$ 且 $L​$ 的行向量都是单位正交向量。因此接下来的讨论里，我们假设 $L​$ 的行向量都是单位正交的。

我们对 $L$ 的行向量进行正交扩充，变成 $\mathbf l_1, ..., \mathbf l_k, \mathbf l_{k+1}, ...,\mathbf  l_n$，使其成为 $N$ 维的一个单位正交基。这时候就可以把 $x_i$ 表示成 $\mathbf x_i = x_{i1} \mathbf l_1 + ... +x_ik\mathbf l_k + x_{i,k+1} \mathbf l_{k+1} + ... +x_{in} \mathbf l_n$。

这时候 $L\mathbf y_i = y_{i1}\mathbf l_1 + ... + y_{ik}\mathbf l_k​$ 很显然可以看出，如果要让Loss最小，必须有 $x_{i1} = y_{i1}, x_{i2} = y_{i2} ,...., x_{ik} = y_{ik}​$ 。于是 $\mathbf x_i - L\mathbf y_i​$ 可以化作 $ x_{i, k+1}\mathbf l_{k+1} + ... + x_{i, n}\mathbf l_n​$。

## 最小化目标函数

 $ x_{i, k+1}\mathbf l_{k+1} + ... + x_{i, n}\mathbf l_n ​$ 可以表示为 $R^TR \mathbf x​$, 其中 $R = \begin{bmatrix} \mathbf l_{k+1} \\ ... \\ \mathbf l_n\end{bmatrix}​$

于是损失函数变成 $\sum\limits_i (R^TR\mathbf x)^T(R^TR\mathbf x) = \text{tr}(X^TR^TRR^TRX) = \text{tr}(X^TR^TRX)= \text{tr}(RXX^TR^T)$ 

(由于$RR^T$ 显然是 $(n-k) \times (n-k)$ 的恒等矩阵)

 由于$ XX^T​$ 是半正定矩阵，因此我们可以把其分解为 $XX^T = P^{-1} \Lambda P = P^T \Lambda P​$ ，其中$P​$ 是单位正交阵，每一行是一个 $XX^T​$ 的一个单位特征向量。因此 $P = \begin{bmatrix} \mathbf e_1 \\ ... \\ \mathbf e_n\end{bmatrix}​$ 。我们这里假定$\Lambda = \begin{bmatrix} \lambda_1 & ... & ... \\ ... & \lambda_2 & ... \\ ...& ... & ...\end{bmatrix}​$ ，$\lambda_1, \lambda_2 ...​$ 按照递减序排 列。

由于 $XX^T​$ 是半正定矩阵，因此这些特征值都是非负的。

此时我们就可以把损失函数化为 

$\sum \limits_{i = 1}^n \sum\limits_{j = k+1}^n \lambda_i (\mathbf e_i \cdot \mathbf l_j)^2 =   \\
\lambda_1 (\mathbf (e_1\cdot\mathbf l_{k+1})^2 + ... + (\mathbf e_1\cdot \mathbf l_{n-1})^2  + (\mathbf e_1\cdot \mathbf l_n)^2) + \\
...+\\
\lambda_{n-1}((\mathbf e_{n-1} \cdot \mathbf l_{k+1})^2+...+(\mathbf e_{n-1} \cdot \mathbf l_{n-1})^2+(\mathbf e_{n-1} \cdot \mathbf l_{n-1})^2)+\\
\lambda_n((\mathbf e_n \cdot \mathbf l_{k+1})^2+...+(\mathbf e_n \cdot \mathbf l_{n-1})^2+(\mathbf e_n \cdot \mathbf l_n)^2)​$

上式记作 **(1)** 式

其中有 $(\mathbf e_1 \cdot \mathbf l_i)^2 + ... + (\mathbf e_n \cdot \mathbf l_i)^2 = 1​$ 对于任何 $i = k+1, k+2, ..., n​$

同时又有 $(\mathbf e_i \cdot \mathbf l_{k+1})^2 + ... + (\mathbf e_i \cdot \mathbf l_n)^2 = 1 - ((\mathbf e_i * \mathbf l_1)^2 + ... + (\mathbf e_i * \mathbf l_k)^2)\le 1$。

可见为了让整个损失函数尽可能小，要尽可能把权重（$\mathbf e_i \cdot \mathbf l_j​$）集中到后面的行上。仅仅在满足上面的约束条件的情况下，其最小值也只可能达到$\lambda_{k+1} + ... + \lambda_n​$，因为如果在 $i \le k​$ 的某行上有一个 $(\mathbf e_i \cdot \mathbf l_j)​$，总是可以将其的值沿着列移动到更下面的行和未达到1的行上。

又可以看出来，当 $\mathbf l_j = \mathbf e_j​$ 的时候，损失函数满足了最小值 $\lambda_{k+1} + ... + \lambda_n​$。为了让(1)式达到最小值，我们需要

$(\mathbf e_{n-1} \cdot \mathbf l_{k+1})^2+...+(\mathbf e_{n-1} \cdot \mathbf l_{n-1})^2+(\mathbf e_{n-1} \cdot \mathbf l_{n-1})^2 = 1 \\ ... \\(\mathbf e_n \cdot \mathbf l_{k+1})^2+...+(\mathbf e_n \cdot \mathbf l_{n-1})^2+(\mathbf e_n \cdot \mathbf l_n)^2 = 1$

又因为 $\mathbf e_j ,\mathbf l_i$ 都是单位正交向量，所以可以得到，$\mathbf l_{k+1}, ... \mathbf l_n$ 都可以用 $\mathbf e_{k+1}, ...,\mathbf e_{n}$ 线性表示，或者可以说$\{\mathbf l_{k+1}, ... \mathbf l_n\}$ 和 $\{\mathbf e_{k+1}, ...,\mathbf e_{n}\}$ 是 $n$ 维线性空间下的某一个 $n-k$ 维子空间的两组单位正交基。

## 结论

通过分析可以看出，如果要舍弃 $k < n$ 个维度，则其的误差最小就是 $XX^T$ 的最小的 $k$ 个特征值的和。

此时 $\mathbf x - Ly = R^TR\mathbf x$ 于是，$Ly = (I - R^TR)\mathbf x = L^TL\mathbf x$ 。这一步成立是因为 $L$ 的行向量是单位正交基

$\mathbf l_1,..., \mathbf l_n$ 的前 $k$ 个，$R$ 的行向量则是其后 $n - k$ 个，我们可以把 $L$ 从 $k \times n$ 增广到 $n \times n$ 变成 $\begin{bmatrix}L\\O\end{bmatrix}$，同理把 $R$ 增广到 $\begin{bmatrix}O\\R\end{bmatrix}$ ，则 $L+R$ 就是单位正交阵，很容易推得上面的结论。

为了达到最小误差，我们只需要保证 $L$ 矩阵的行向量可以用$XX^T$ 的前 $k$ 个特征向量表示即可。一般情况为了方便，直接使得 $\mathbf l_1 = \mathbf e_1, ..., \mathbf l_k = \mathbf e_k​$，这样就得到了常用的PCA公式。