## Seq2Seq的基本原理

论文《Sequence to Sequence Learning with Neural Networks》[1]阐述了使用循环神经网络进行序列到序列的映射方法。该论文的思想也很简单：

> ​	A simple strategy for general sequence learning is to map the input sequence to a ﬁxed-sized vector using one RNN, and then to map the vector to the target sequence with another RNN (this approach has also been taken by Cho et al. [5]).

也就是说可以把输入序列的RNN（RNN-Encoder）的最后的隐层状态作为一个反应输入序列的向量，然后通过线性变换把该向量映射为输出序列的RNN（RNN-Decoder）的初始隐层状态，然后通过输出的RNN来产生目标序列。

![rnn_seq2seq](D:\MyDoc\文章\Learn\figs\rnn_seq2seq.PNG)

模型的结构如图所示，其中左边是 RNN Encoder，右边是 RNN Decoder，其中 go symbol 是某一个固定的输入，表示输出序列的开始。

在训练的时候，解码器并不把 t-1时刻的输出作为t时刻的输入，而是直接把目标序列的t-1时刻的值作为t时刻的输出，以提高训练效率。

## 使用Tensorflow做一个简单的Seq2Seq模型

貌似现在tensorflow的很多API都要转到tf.keras上面去，因此我决定用keras实现这个模型。

首先创建一个RnnSeq2Seq类，继承keras.layers.Layer。根据[Keras官方文档](https://keras.io/layers/writing-your-own-keras-layers/)，一般在 `__init__` 中，实现一些与Layer输入无关的变量的初始化，在 `build()` 中，则是和输入有关的变量的初始化（因此 build 函数会有个input_shape 参数）

比如 `cell = SimpleRNNCell(units=10)` 这行代码，执行了`__init(self)__` 函数，其实里面并没有任何变量的初始化，只有在 `output, next_state = cell(input, state)` 这行代码中，才会执行 `cell.build(input_shape)`真正初始化变量，因为在这个时候 这个RNN cell才知道输入的维度，比如输入的向量大小是[1, 10]，且初始化的时候 units = 20，那么就可以产生一个[10, 20] 的矩阵 W_input，一个[20, 20]的矩阵 W_state，已经一个[20]的向量 Bias。查看 tensorflow\python\keras\_impl\keras\layers\recurrent.py 里面的 SimpleRNNCell 源代码可以印证：

```python
@shape_type_conversion
  def build(self, input_shape):
    self.kernel = self.add_weight(
        shape=(input_shape[-1], self.units),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.units,),
          name='bias',
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True
```

不过由于自己的代码写的比较混乱，所以就默认一开始就知道输入序列的长度，在 `__init__` 里面就初始化变量了。

```python
class RnnSeq2Seq(tf.keras.layers.Layer):
    def __init__(self, hidden_size, encoder_rnn_cell: tf.keras.layers.Layer, decoder_rnn_cell: tf.keras.layers.Layer,
                 pred_length, go_symbol):
        super(RnnSeq2Seq, self).__init__()
        self.weight = self.add_variable("kernel", shape=[hidden_size, hidden_size],
initializer=tf.keras.initializers.glorot_normal())
        self.bias = self.add_variable("bias", shape=[hidden_size], 					initializer=tf.keras.initializers.glorot_normal())
        self.weight_o = self.add_variable("weight_o", shape=[hidden_size, 1],
                                       initializer=tf.keras.initializers.glorot_normal())
        self.bias_o = self.add_variable("bias_o", shape=[1], initializer=tf.keras.initializers.glorot_normal())
        self.encoder_rnn_cell = encoder_rnn_cell
        self.encoder_rnn = tf.keras.layers.RNN(self.encoder_rnn_cell)
        self.decoder_rnn_cell = decoder_rnn_cell
        self.pred_length = pred_length
        self.go_symbol = go_symbol
        self.training = tf.keras.backend.variable(True, dtype=bool)
```

通过代码可以看到，自己定义的 RnnSeq2Seq 层有如下几个变量：

1. encoder_rnn_cell: 用来作为 RNN Encoder
2. decoder_rnn_cell: 用来作为 RNN Decoder
3. kernel bias，这两个变量就是一个单层神经网络的Weight和Bias，把 RNN encoder 末状态的隐层向量映射到 RNN decoder 的初始状态。相当于  $ h_{decoder,0} = \sigma(h_{encoder,len-1} \cdot kernel + bias)$
4. weight_o, bias_o，这两个变量把RNN cell的隐藏层状态转换为输出，因为普通的RNN的output默认就是state，所以还需要做转换。
5. training，这个变量用来标识是否处于训练模式。

然后再看 build 函数

```python
    def build(self, input_shape):
        super(RnnSeq2Seq, self).build(input_shape)
        # print(input_shape, self.compute_output_shape(input_shape))
        self.encoder_rnn_cell.build(input_shape[0][1:])
        self.decoder_rnn_cell.build(input_shape[1][1:])
```

这里必须对 rnn_cell 执行 build（同时计算好 input_shape）。

*这里如果不对rnn_cell build就会报错`'SimpleRNNCell' object has no attribute 'kernel'` 大概原因是rnn_cell 并不能直接当成函数使用，而是要调用 rnn_cell.call(inputs, state) 方法。否则会报错`...only takes 2 positional args but 3 were given`。这里我也参考了keras源码里面的RNN class*

可以注意到RnnSeq2Seq的输入是一个含有两个元素的List，其中一个元素是 源序列，另一个元素是目标序列，它们的维度都是 [batch_size, seq_length, feature_size]。所以我们让 rnn_cell 的 input_shape 变为 [seq_length, feature_size] （似乎只需要最后一维是 feature_size) 就可以，不影响 rnn_cell 的运行。

最主要的逻辑实在 `call` 函数里完成的。

```python
    def call(self, inputs, **kwargs):
        states = self.encoder_rnn(inputs[0])
        final_state = states
        state = [tf.keras.activations.tanh(tf.keras.backend.dot(final_state, self.weight) + self.bias)]
        prev_output = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(tf.zeros_like(inputs[1]))
        pred_outputs = [prev_output]
        for i in range(self.pred_length):
            if i == 0:
                prev_output = pred_outputs[-1]
            else:
                prev_output = tf.cond(self.training,
                                      lambda: tf.keras.layers.Lambda(lambda x: x[:, i - 1, :])(inputs[1]),
                                      lambda: pred_outputs[-1])
            output, state = self.decoder_rnn_cell.call(prev_output, state)
            pred_outputs.append(tf.keras.activations.tanh(tf.keras.backend.dot(output, self.weight_o) + self.bias_o))
        return tf.keras.backend.permute_dimensions(tf.keras.backend.stack(pred_outputs[1:]), [1, 0, 2])
```

这里encoder的最后状态直接使用 encoder_rnn 得出（直接使用 tf.keras.layers.RNN)，注意RNN的返回值有好几种情况，可以直接查看keras的源代码。这里面只返回最后一个时刻的隐藏层状态。

接下来就是用一个神经网络映射隐层状态，然后作为解码的RNN的则通过循环实现（因为默认的RNN并不能把上一时刻输出作为当前的输入）。这里我们让第一个输入为0向量，注意这里用了 Lambda Layer 来产生和某个时刻的输入相同形状的0向量([batch_size, input_feature])。而且这里要用到 `self.training` 来判断使用真实的目标序列作为输入或是使用上一个输出作为输入。**由于Keras似乎没有自带控制流语句，所以只能使用tf.cond来进行控制流操作。**

由于SimpleRNNCell的输出就是隐层状态，所以仍需要进行变换得到序列下一项的预测值：

`pred_outputs.append(tf.keras.activations.tanh(tf.keras.backend.dot(output, self.weight_o) + self.bias_o))`

注意最后的返回值，因为`pred_outputs` 是一个长度为 `seq_length` 的数组，里面每一项是 `[batch_size, feature_size]` 的tensorflow，stack之后shape为 `[seq_length, batch_size, feature_size]`，所以需要permute_dimensions ，交换第0维和第1维变成 `[batch_size, seq_length, feature_size]` 的形状。注意不能用`reshape`。如：一个 2×3的数组

$a = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}$

如果 reshape 成 3 × 2 的，就变成了

$\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6\end{bmatrix}$

而permute_dimension([1, 0])，则变成了

$\begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6\end{bmatrix}$

所以这里我们希望使用的是permute_dimension。

如下的Class把keras.models.Model封装起来。

```python
class RnnSeq2SeqModel:
    def __init__(self, src_seq_len: int, target_seq_len: int, rnn_seq2seq: RnnSeq2Seq):
        self.src_seq_len = src_seq_len
        self.target_seq_len = target_seq_len
        self.input_seqs = tf.keras.layers.Input([src_seq_len, 1])
        self.target_seqs = tf.keras.layers.Input([target_seq_len, 1])
        self.rnn_seq2seq = rnn_seq2seq
        self.output = self.rnn_seq2seq([self.input_seqs, self.target_seqs])
        self.model = tf.keras.models.Model(inputs=[self.input_seqs, self.target_seqs], outputs=self.output)

    def train(self, dataset: Dataset, batch_size=30, epochs=100, learning_rate=0.1):
        tf.keras.backend.set_value(self.rnn_seq2seq.training, True)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate), loss="mean_squared_error", metrics=["mean_squared_error"])
        print(self.model.trainable_weights)
        all_data = dataset.get_all_train_data()
        self.model.fit([all_data[:, :self.src_seq_len].reshape([-1, 10, 1]), all_data[:, self.src_seq_len:].reshape([-1, 10, 1])] , all_data[:, self.src_seq_len:].reshape([-1, 10, 1]), batch_size, epochs)

    def validate(self, dataset:Dataset):
        tf.keras.backend.set_value(self.rnn_seq2seq.training, False)
        test_data = dataset.get_test_data()
        loss = self.model.evaluate([test_data[:, :self.src_seq_len].reshape([-1, 10, 1]), test_data[:, self.src_seq_len:].reshape([-1, 10, 1])],
                            test_data[:, self.src_seq_len:].reshape([-1, 10, 1]))
        print(loss)
```

最后实验的时候，我用的是 正弦函数产生的序列，源序列和目标序列都是正弦曲线，但是有相位的差异。使用了900个数据点进行训练，100个进行测试，epochs=100。通过多次尝试，我发现 `batch_size = 10, learning_rate=0.3` 的训练效果较好，如图：

![seq2seq_plots](D:\MyDoc\文章\Learn\figs\seq2seq_plots.jpg)

其中蓝色表示源序列，绿色表示目标序列，橙色表示预测序列。可见预测的序列还是比较准确的。

*为了确保代码没有写错，再进行测试的时候直接把源序列当做目标序列作为输入（理论上测试的时候解码器是根据上一时刻的输出作为下一时刻的输入，所以不需要使用目标序列的任何数据）。一定要注意在神经网络里实现控制流，并不能直接用 if-else，因为神经网络运行的时候并不是按照代码执行的，写的代码其实只是在构建网络的结构，所以即使使用了if-else，也仅仅是在构造结构的时候用到。一开始我直接用了一个bool变量作为 training的标记，结果发现那样网络就一直在训练模式，因为初始化网络的时候这个变量的值是True*

## 引用文献

[1] Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[C]//Advances in neural information processing systems. 2014: 3104-3112.