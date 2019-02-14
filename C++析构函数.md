# C++移动构造函数的分析

在设计一个矩阵类的时候，我们需要定义一些操作符，比如加法：

```c++
Matrix operator+(const Matrix &mat_1, const Matrix &mat_2)
{
	Matrix sum(mat_1.NRow(), mat_1.NColumn());
	for (int i = 0; i < sum.NRow(); i++)
		for (int j = 0; j < sum.NColumn(); j++)
			sum(i, j) = mat_1(i, j) + mat_2(i, j);
	return sum;
}
```

这样子定义加法，可以看到，作为结果的变量 `Matrix sum` 是在函数里面的局部变量，在函数返回的时候，局部变量就会执行销毁操作，所以在函数执行结束后会调用 `Matrix` 的析构函数。

根据一开始定义的 Matrix 构造函数

```C++
Matrix(const Matrix &mat) : nRow{ mat.nRow }, nColumn{ mat.nColumn }, data{ new float[mat.nRow * mat.nColumn] }
{
    for (int i = 0; i < nRow * nColumn; i++)
    {
        data[i] = mat.data[i];
    }
}
```

可以看出，在定义一个Matrix的时候，会产生拷贝操作。因此，对于函数的返回值，实际上也会进行拷贝操作，也就是说，对于 `operator+()` ，在返回的时候，依然要把内部局部变量的值拷贝到函数外部，损失很大性能。

为此，可以定义移动构造函数，这个构造函数可以更改他的参数，如下：

```c++
Matrix(Matrix &&mat) : nRow{ mat.nRow }, nColumn{ mat.nColumn }, data{ mat.data } 
{
    mat.data = nullptr;
    mat.nRow = 0; mat.nColumn = 0;
}
```

这样子就不需要进行拷贝操作，只需要把指针从一个Matrix移动到另一个。当定义了移动构造函数，则函数返回时就会默认使用这个函数，从而不需要进行拷贝操作。