# FlexNeuroNet

## 简介

这是我用C++写的一个简单的神经网络框架。

##示例

### NeuronFactory API

```c++
	Dataset dataset; //数据集
	dataset.ReadFromCSV("data.csv", 2, false); // 从CSV中读入数据
	Model model; //模型
	//创建神经元
	Neuron * input1 = model.nf()->MakeInputNeuron(); 
	Neuron * input2 = model.nf()->MakeInputNeuron();
	Neuron * pred = model.nf()->MakeNeuron(Activations.sigmoid);
	Neuron * label = model.nf()->MakeInputNeuron();
	// 平方损失函数 (y_pred - y_true)^2
	Neuron * loss = model.nf()->MakeNeuron(Activations.square);
	// Loss 神经元不能被训练
	loss->SetTrainable(false);
	// 下面两行对应 w1x1+w2x2
	model.nf()->MakeSynapsis(input1, pred, true, 0.5);
	model.nf()->MakeSynapsis(input2, pred, true, 1.5);
	// 对应 y_pred - y_true
	model.nf()->MakeConstantSynapsis(pred, loss, 1);
	model.nf()->MakeConstantSynapsis(label, loss, -1);
	// 最小化 Loss 神经元的输出
	model.SetTargetNeuron(loss);
	model.SetData(&dataset);
	model.Bind(input1, 0);
	model.Bind(input2, 1);
	model.Bind(label, 2);
	// 普通的变量初始化（ 1/n_in )
	Initializer initializer;
	model.Initialize(&initializer);

	for (size_t i = 0; i < 10000; i++)
	{
		model.TrainBatch(10, 0.1);
		if(i % 10 == 0)
			model.ValidateBatch(10);
	}
```

###LayerFactory API and Initializer

```c++
	Dataset dataset;
	dataset.ReadFromCSV("Data/data.csv", 2, false);
	Model model;
	Layer * input = model.lf()->MakeInputLayer(2);
	Layer * pred = model.lf()->ConnectFrom(input, 1, Activations.sigmoid);
	Layer * label = model.lf()->MakeInputLayer(1);
	Layer * loss = model.lf()->LinearFromTwoLayers(pred, label, 1, -1, Activations.square);
	loss->SetTrainable(false);
	model.SetTargetNeuron(loss->Neuron(0));
	model.SetData(&dataset);
	model.BindLayer(input, 0);
	model.BindLayer(label, 2);
	Initializer initializer;
	model.Initialize(&initializer);
```

```c++

```

