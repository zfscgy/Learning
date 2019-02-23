### Datagrid的刷新问题

某个Datagrid的两列之间有函数关系，譬如我希望在修改A列的时候，B列的值也跟随A列进行变化。这时候我使用了 CellEditEnding，代码如下：

```c#
if((e.Column.Header as string) == "原单-高")
{
    try
    {
        (e.Row.Item as OrderUnit).CoverH = float.Parse((e.EditingElement as TextBox).Text) + client.DiffH;
        Datagrid_Units.Dispatcher.BeginInvoke(new                                              		Action(()=>Datagrid_Units.Items.Refresh()),                                             System.Windows.Threading.DispatcherPriority.Background);
    }
    catch { }
}
```

注意，如果直接使用 `Datagrid.Items.Refresh()`是无效的，因为此时正在Editing，是无法刷新数据的。

但是即便如此，如果在编辑完成之后用鼠标点击另外的单元格，仍然会出现问题。因为此时依然在Editing Mode, 导致`Refresh()`失败。

经过我的尝试，发现通过如下代码：

```c#
 ((TextBlock)Datagrid_Units.Columns[5].GetCellContent(e.Row)).Text = CoverH.ToString();
```

可以成功解决这个问题。（通过调试发现GetCellContent在非编辑模式下是TextBlock）

原理大约是这样子是直接修改了UI元素，而在Editing Mode，UI元素并没有锁定，只是不允许刷新ItemSource。