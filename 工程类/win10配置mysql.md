# Win10配置MySql8.0

安装完MySql Server8.0，按照网上的说法，首先配置一个 my.ini 文件，如下

```mysql
[client]
port=3306
[mysqld]

basedir="C:\Program Files\MySQL\MySQL Server 8.0"
datadir="C:\Program Files\MySQL\MySQL Server 8.0\data"
skip-grant-tables
bind-address=127.0.0.1
port=3306
```

这个文件其实就像是运行 mysqld.exe 时的参数，比如 `skip-grant-tables`就对应运行时加入`--skip-grant-tables`，表示不检查权限表，即任何人都可以登录数据库。后面两项则表示这个服务会运行在"127.0.0.1:3306"地址，即本机的3306端口。

然后先初始化数据库，采用的是 ` mysqld --initialize --user=mysql --console`

根据官方文档，这是初始化数据库的"data"目录，就是初始化一些内置的数据库。否则无法启动数据库。当执行完这个语句后，会分配给root用户一个默认的密码。

`--console` 命令是为了让错误信息直接在命令行里显示。

`--user` 选项表示执行命令的“系统用户”名称，在这里似乎填写任意用户名都可以。

然后需要安装mysql服mysqld --install mysql --defaults-file="C:\Program Files\MySQL\MySQL Server 8.0\bin\my.ini"务，采用的方法是

`mysqld --install mysql --defaults-file="C:\Program Files\MySQL\MySQL Server 8.0\bin\my.ini"`

根据文档，`--install`后面跟的是服务名，可以任取，我们就让他为`mysql`。

> [`--install  [*service_name*\]`](https://dev.mysql.com/doc/refman/8.0/en/server-options.html#option_mysqld_install)



注意最后用了一个长长的绝对路径，是因为如果直接用`--defaults-file=my.ini`很可能会找不到文件，除非加入了系统环境变量。

这时候使用`net start mysql`，就会显示服务已经成功启动。

## Can't connect to MySQL server on 'localhost'

然而在命令行输入`mysql`，却发现出现了`Can't connect to MySQL server on 'localhost'` 报错信息，但是查看任务管理器-服务，却发现mysql服务明明在运行。这时候查看安装目录下的 data 文件夹，可以看到一个后缀名为.err的文件。打开可以看到如下的报错信息：

> 2018-09-13T02:07:54.423068Z 0 [ERROR][MY-010131] [Server] TCP/IP, --shared-memory, or --named-pipe should be configured on NT OS
> 2018-09-13T02:07:54.423571Z 0 [ERROR][MY-010119] [Server] Aborting
> 2018-09-13T02:07:54.606308Z 0 [Warning][MY-011311] [Server] Plugin mysqlx reported: 'All I/O interfaces are disabled, X Protocol won't be accessible'

于是怀疑mysql并没有成功地设置好网络，于是看命令行netstat -ano，发现的确3306端口没有占用。

怀疑是防火墙的问题，把防火墙关闭，或者新建规则，却依然不能连接。在网上搜索了大量时间后，才发现是`skip-grant-tables`的问题。在官方文档里，有：

>If the server is started with the [`--skip-grant-tables`](https://dev.mysql.com/doc/refman/8.0/en/server-options.html#option_mysqld_skip-grant-tables) option to disable authentication checks, the server enables [`--skip-networking`](https://dev.mysql.com/doc/refman/8.0/en/server-options.html#option_mysqld_skip-networking) automatically to prevent remote connections.

好像是MySql8.0版本才加入了这个特性。

于是把`skip-grant-tables` 从my.ini去掉，或者加上`shared-memory`，可以解决这个问题。注意当`skip-grant-tables`去掉后，就必须记住生成的随机密码，然后要新建数据库，就需要`alter user`来更改密码。

