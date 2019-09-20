# EM算法的个人理解和实验

## 引言



##EM算法的通俗解释

Expection Maximization 的算法的本质，就是对一组数据 $\{x_1, x_2, x_3, ...\}$，假设他们满足某种分布 $D(\theta)$ , 求出这个最优的 $\theta$ , 使得 $\prod p(x_i)$ 的值最大。在不存在隐变量的情况下，也就是说，$ p(x|\theta) = f(x, \theta) $ ，则这个问题就是个简单的求函数极值的问题，也就是最普通的参数估计问题，相信大家概率论的课上都学过了。

但是在实际情况下，除去分布的参数 $\theta$， 还有一些隐变量 $h$ ，也决定了 $ x$ 的分布。一个简单的例子就是对于很多数据点 $ \{x_1, x_2, x_3, ...\}$ , 其中的任一个点都是两个正态分布 $ N(\mu_1,\sigma_1^2) , N(\mu_2, \sigma_2^2)$ 中的其中一个。那么每一个 $x_i$ 属于哪一个正态分布，产生的标签 $c_i \in \{0, 1\}$ 就是隐变量。

但是我们是不是也可以把隐变量当做分布的参数呢？也就是说从两个正态分布里面取样产生的新的分布的参数包含$\{ \mu_1, \sigma_1, \mu_2, \sigma_2, c_1, c_2, c_3, ....\}​$ ，搜索整个参数空间，理论上也可以找到一组最佳的参数，使得 $p(x|h, \theta)​$ 最大。

但是这样就和我们的目标有所违背：我们的目标是找到 $p(x|\theta)​$ 的最大值，并不是 $p(x|\theta, h)​$ 的最大值。而且一般 $p(x|\theta, h)​$ 的函数也十分复杂，很可能无法求出极值。

通过贝叶斯法则，我们有： $p(x|\theta) = \dfrac{p(x, h|\theta)}{p(h|x,\theta)}​$ ，但是这个值和 $h%​$ 有关，所以很难直接对这个公式下手。EM算法提供了一种巧妙的方法：

对上面的等式求对数，得到: $ \log p(x|\theta) = \log p(x,h|\theta) - \log p(h|x,\theta)​$ , 同时又有 $ \sum\limits_h p(h|x,  \theta) = 1​$ , 因此把这个对 $h​$ 加权求和的算符作用在左右两边，很显然有：$ \log p(x|\theta) = \sum\limits_h p(h|x,\theta) \log p(x, h|\theta) - \sum\limits_h p(h|x,\theta) \log p(h|x,\theta)​$  , 出现了很熟悉的 $p \log p​$ 的形式！貌似可以利用信息论里面的那些不等式求极值了！

记住左边求和项 $ \sum\limits_h p(h|x,  \theta) = 1$ ，对于任何 $\theta$ 都成立。因为等式左边不含 $h$ , 于是我们可以把上式改成 $ \log p(x|\theta_t) = \sum\limits_h p(h|x,\theta_t) \log p(x, h|\theta_t) - \sum\limits_h p(h|x,\theta_t) \log p(h|x,\theta_t) $ 。观察等号左边的第二项， $- \sum\limits_h p(h|x,\theta_t) \log p(h|x,\theta_t) $ 就是在条件 $x,\theta$ 下 $h$ 的信息熵。信息论告诉我们信息熵是最短编码长度。那么如果变量的分布发生改变，但编码方式不变（也就是说左边的 $p(h|x,\theta)$ 变化了），必然会增加平均码长。也就是说 $- \sum\limits_h p(h|x,\theta_t \log p(h|x,\theta) >  - \sum\limits_h p(h|x,\theta_t) \log p(h|x,\theta) $ 。当然其实这本质就是“相对熵必然为正”的 Gibbs不等式。

这告诉我们，把$ \log p(x|\theta_t) = \sum\limits_h p(h|x,\theta_t) \log p(x, h|\theta_t) - \sum\limits_h p(h|x,\theta_t) \log p(h|x,\theta_t)$  右边的 $\theta_t$ 换成 $\theta$ ，有$ \log p(x|\theta) = \sum\limits_h p(h|x,\theta_t) \log p(x, h|\theta) - \sum\limits_h p(h|x,\theta_t) \log p(h|x,\theta)$ ， 等号右边的第二项是一定增加的，所以我们只要保证 $\sum\limits_h p(h|x,\theta_t) \log p(x, h|\theta)$ 增加就行。

这就是EM算法的精髓，通过对于当前的$\theta_t​$ ，找到一个$\theta​$， 使得  $\sum\limits_h p(h|x,\theta_t) \log p(x, h|\theta)​$  取到最大值，不断迭代，最终找到最优的 $\theta​$ 。

## 一个简单的例子

就用上文的两个正态分布的例子，对于数据点$\{x_1, x_2, ... ,x_n\}​$，他们属于两个正态分布之一，参数分别为$\{\mu_1, \sigma_1, \mu_2, \sigma_2\}​$ ，数据的标签为$\{c_1,c_2,...,c_n\}​$。$c_i = 1​$ 表示 $x_i \sim N(\mu_1, \sigma_1)​$，$c_i = 2​$ 表示 $x_i \sim N(\mu_2, \sigma_2)​$。这时候我们把数据的标签作为隐变量。

假设目前已经进行到 t 步，即当前的参数分别为 $\{\mu_{1,t}, \sigma_{1,t}, \mu_{2,t}, \sigma_{2,t}\}​$, 此时有我们要求出优化目标 $\sum\limits_h p(h|x,\theta_t) \log p(x, h|\theta)​$  求和号里面的左边部分 $p(h|x,\theta_t) = \dfrac{p(h,x,\theta_t)}{p(x,\theta_t)} = \dfrac{p(h, x|\theta_t)}{p(x|\theta_t)}​$  。

这个式子依然难以计算，但是注意到

$p(h,x|\theta) = p(x|h,\theta) \cdot p(h|\theta)$， 左边的 $p(x|h,\theta)$ 一般都是有函数表达式的，右边的 $p(h|\theta)$ ，在这里我们认为隐变量 $h$ 和参数 $\theta$ 是无关的，并且认为 $h$ 的分布由另一组参数 $\Tau$ 决定，即：$p(h|\theta) = p(h)$。在这里，我们很容易认为单个数据的标签满足伯努利分布，即：有 $\tau$ 的概率标签为 1，表示该数据点属于第一个正态分布；反之则属于第二个。所以 $\tau$ 也要加入参数中，参数集合变为：$\{\mu_1, \sigma_1, \mu_2, \sigma_2, \tau\}$

于是我们有： $p(x, h|\theta) = p(x|h,\theta) * p(h)​$  ，

$\sum\limits_h p(h|x,\theta_t) \log p(x, h|\theta) = \sum\limits_h \dfrac{p(h)\cdot p(x|h,\theta_t)}{\sum\limits_h p(h) \cdot p(x|h,\theta_t)} \log p(x|h, \theta)\cdot p(h)​$  

对于任何一个数据点 $x_i$ ，对于隐变量 $h$, 只需要考虑 $h_i$ 的值，因为别的数据点处于哪一个分布对$p(x_i|h,\theta)$ 没有影响。因此可以把隐变量的集合 $\{h\}$ 拆成 $H_1 = \{h|h_i = 1\},\ H_2 = \{h|h_i = 2\}$ 两个集合。这样我们就可以把公式拆成：

$\dfrac{1}{\sum\limits_{h\in H_1} p(h) p(x_i|h_i = 1, \theta_t) +\sum\limits_{h\in H_2}{p(h)p(x_i|h_i=2, \theta_t)}} \cdot \\ (\sum\limits_{h \in H_1} p(h) p(x_i|h_i = 1, \theta_t) \log p(h) p(x_i|h_i = 1, \theta)   + \sum\limits_{h\in H_2} p(h) p(x_i|h_i = 2, \theta_t) \log p(h) p(x_i|h_i = 2, \theta))$  

显然有：$\sum\limits_{h\in H_1} p(h) = \tau,\ \sum\limits_{h\in H_2} p(h) = 1-\tau​$ 

最后我们需要考虑的是，上面的讨论仅仅是最大化某一个样本的 $\log p(x|\theta)​$ 的值的，考虑到我们有大量的样本，实际的优化目标 $p(X|\theta) = \prod\limits_{x \in X} p(x|\theta) ​$ ，取对数后相加即可。因此在上式的左侧还要加上 $\sum\limits_{x_i \in X}​$

于是最终的式子为：

$\sum\limits_{x_i \in X} \sum\limits_{h_i \in \{1,2\} } p(h_i|x_i, \theta_t) \log \tau_{h_i} p(x_i|h_i, \theta)​$ 

其中，$\tau_1 = \tau,\  \tau_2 = 1 - \tau,\ p(h_i|x_i, \theta_t) = \dfrac{\tau_{h_i}\cdot p(x_i|h_i, \theta_t)}{\sum\limits_{h_j \in \{1, 2\}}\tau_{h_j}\cdot p(x_i|h_j, \theta_t)} ​$

考虑到上式的右端 $\log \tau_{h_i}p(x_i|h_i, \theta)​$ 中的 $\tau_{h_i}​$ 与 $\theta​$ 无关，因此可以忽略之。再看这个式子，是不是很像

$\sum\limits_{x \in X} w_x \log p(x|\theta)​$ 的形式？因为这是对数形式，我们还原出其原始式子：$\prod\limits_{x \in X} p(x|\theta)^{w_x}​$ 很显然这就是一个概率值，可以理解为 $x​$ 样本出现了 $w_x​$ 次数的概率。注意到不同的 $h_i​$ 对应的 $x​$ 属于不同的正态分布，所以我们要把关于 $ h_i ​$ 的那个求和的两项分别拿出来求极值。

拿出 $h_i = 1​$ 这一项，$p(x_i|h_i,\theta_t) = \dfrac{1}{\sqrt{2\pi}\sigma_1} e^{-\dfrac{(x_i - \mu_1)}{2\sigma_1^2}} = f(x_i;\mu_1,\sigma_1)​$ 。因此这就是个正态分布的加权最大似然估计（Weighted MLE for Normal Distribution），很容易可以得到：

$ \mu_{1,t} = \dfrac{\sum\limits_i\dfrac{\tau_1 \cdot f(x_i;\mu_1, \sigma_1)}{\tau_1 \cdot f(x_i;\mu_1, \sigma_1) + \tau_2 \cdot f(x_i;\mu_2, \sigma_2)} \cdot x_i}{\sum\limits_i\dfrac{\tau_1 \cdot f(x_i;\mu_1, \sigma_1)}{\tau_1 \cdot f(x_i;\mu_1, \sigma_1) + \tau_2}} = \dfrac{\sum\limits_i T_{i,1} x_i}{\sum\limits_i T_{i, 1}}​$

其中，$T_{i,j} = \dfrac{\tau_j \cdot f(x_i;\mu_j, \sigma_j)}{\tau_1 \cdot f(x_i;\mu_1, \sigma_1) + \tau_2 \cdot f(x_i;\mu_2, \sigma_2)}\quad j \in \{1,2\}​$

$\sigma_{1,t}^2 = \dfrac{\sum\limits_i T_{i,1}(x_i - \mu_{1,t})^2}{\sum\limits_i T_{i, 1}} $

当然同时，我们还要注意到 $\tau$ 也是一个参数，因此也需要对此进行更新。此时可以把 $p(h_i|x_i, \theta_t) \log \tau_{h_i} p(x_i|h_i, \theta)$ 中的 $\log$ 里的 $p(x_i|h_i, \theta)$ 看做常数。此时形式变成了

$w_1\log(\tau_1) + w_2\log(1-\tau_1)$ 或者是 $\tau_1^{w_1} (1-\tau_1)^{w_2}$ 很显然这就是一个二项分布的概率表达式，相当于从一堆红球和黑球中选了若干次，总共有 $w_1$ 次红球， $w_2$ 次黑球，因此其最大似然估计是 $\tau_1 = \dfrac{w_1}{w_1 + w_2},\quad \tau_2 = \dfrac{w_2}{w_1 + w_2}$

这里 $w_1 = \sum\limits_i T_{i,1}$ ，而且可以注意到 $\sum\limits_i T_{i,1} + \sum\limits_i T_{i,2} = 1$

所以有 $ \tau_1  = \sum\limits_i T_{i, 1}$

## 一些理解

从上面的例子中，我们可以看到某个数据点属于哪一个分布，这是一组隐变量，且这组隐变量中的每一个值满足伯努利分布。隐变量自身也包含了参数。因此我们可以画出如下的关系图：

$\theta \rarr h​$  ， $\theta, h \rarr x​$ ，而且 $x​$ 是我们能观测到的数据。EM算法做的事情就是在把 $p(x|\theta)​$ 表示成关于 $\theta​$ 的函数时，找出其最大值。

## 一个简单的实验

```python
import numpy
from matplotlib import pyplot
def em_two_random_normal(mean_1, var_1, mean_2, var_2, size, n_steps = 10):
    x1s = numpy.random.normal(mean_1, var_1, [size, 2])
    x2s = numpy.random.normal(mean_2, var_2, [size, 2])
    xs = numpy.vstack([x1s, x2s])
    mu_1_s = mean_1 + numpy.random.uniform(0, 8, [1, 2])
    mu_2_s = mean_2 + numpy.random.uniform(-8, 0, [1, 2])
    var_1_s = 1
    var_2_s = 1
    p_z1 = 0.5
    p_z2 = 0.5
    mus = [[mu_1_s], [mu_2_s]]
    vars = [[var_1_s], [var_2_s]]
    for _ in range(n_steps):
        p_x_at_para_z1 = numpy.exp(-numpy.sum(numpy.square(xs - mu_1_s), axis=1, keepdims=True) / (2 * var_1_s)) / \
                            numpy.sqrt(2 * numpy.pi * var_1_s)
        p_x_at_para_z2 = numpy.exp( - numpy.sum(numpy.square(xs - mu_2_s), axis=1, keepdims=True) / (2 * var_2_s)) / \
                            numpy.sqrt(2 * numpy.pi * var_2_s)
        p_z1_under_para_x = p_z1 * p_x_at_para_z1 / (p_z1 * p_x_at_para_z1 + p_z2 * p_x_at_para_z2)
        p_z2_under_para_x = p_z2 * p_x_at_para_z2 / (p_z1 * p_x_at_para_z1 + p_z2 * p_x_at_para_z2)
        p_z1 = numpy.mean(p_z1_under_para_x)
        p_z2 = numpy.mean(p_z2_under_para_x)
        mu_1_s = numpy.sum(p_z1_under_para_x * xs, axis=0, keepdims=True) / numpy.sum(p_z1_under_para_x)
        mu_2_s = numpy.sum(p_z2_under_para_x * xs, axis=0, keepdims=True) / numpy.sum(p_z2_under_para_x)
        var_1_s = numpy.sum(p_z1_under_para_x * numpy.sum(numpy.square((xs - mu_1_s)), axis=1, keepdims=True)) / numpy.sum(p_z1_under_para_x)
        var_2_s = numpy.sum(p_z2_under_para_x * numpy.sum(numpy.square((xs - mu_2_s)), axis=1, keepdims=True)) / numpy.sum(p_z2_under_para_x)
        mus[0].append(mu_1_s)
        mus[1].append(mu_2_s)
        vars[0].append(var_1_s)
        vars[1].append(var_2_s)

    pyplot.scatter(xs[:, 0], xs[:, 1])
    for i in range(n_steps):
        pyplot.gca().add_artist(pyplot.Circle([mus[0][i][0, 0], mus[0][i][0, 1]], numpy.sqrt(vars[0][i]), color='red', fill=False))
        pyplot.gca().add_artist(pyplot.Circle([mus[1][i][0, 0], mus[1][i][0, 1]], numpy.sqrt(vars[1][i]), color='blue', fill=False))
    pyplot.xlim(0, 20)
    pyplot.ylim(0, 20)
    pyplot.show()


em_two_random_normal([3,4], 0.5, [12,5], 1.5, 20, n_steps=50)
```

![1556287192787](D:\MyDoc\文章\Learn\figs\EM-visualization.png)

根据图片可以看出，即便我们一开始非常胡乱地设置了初始的参数 $\theta$ ，在几轮迭代之后，EM算法都能很好地找到 $\theta$ 的最大似然估计。