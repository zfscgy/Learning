###系统信息

Ubuntu18.04

###添加用户

为了方便，我们给系统添加一个新的用户，一般可以这样设置

`sudo adduser hadoop`

然后按照提示设置登录密码即可。

### 安装JDK

jdk只需要从Oracle官网下载最新的就好，然后解压缩放在 /opt/之下，同时利用`update-alternatives` 更改默认工具链。

```bash
update-alternatives --install /usr/bin/java java /opt/jdk1.8.0_201/bin/java 100
update-alternatives --install /usr/bin/javac javac /opt/jdk1.8.0_201/bin/javac 100 
```

 `update-alternatives`主要用于管理一个程序有多个版本的情况，比如java有多个版本，通过这样让 /use/bin 中的java链接到 /etc/alternatives 中的java，来实现版本的选择，具体功能其实和环境变量差不多。

### 安装SSH并且在本机上运行

ssh的安装非常简单，`sudo apt-get install ssh-server ssh-client`

然后输入 `ssh-keygen -t rsa`，不要键入passphrase，就可以产生公钥和私钥，分别保存在 ~/.ssh/id_rsa.pub 和 ~/.ssh/id_rsa

为了让本机同时作为客户端和服务器，我们直接把本机的公钥加入 authorized_keys 文件中，这样子就可以实现本机到本机的连接。命令为 `cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys`

测试运行SSH

`ssh localhost`

### 安装HADOOP

直接从Hadoop官网下载压缩包，解压在~/hadoop文件夹里 (/home/hadoop/hadoop)，然后再设置一下环境变量，可以直接在~/.bashrc文件里加入以下语句：

```bash
export HADOOP_HOME=/home/hadoop/hadoop/hadoop-3.2.0
export HADOOP_INSTALL=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export PATH=$PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/native"
```

关于环境变量的设置，在[这篇博文](https://www.cnblogs.com/kevingrace/p/8072860.html)里阐述的比较清楚

>/etc/profile: 此文件为系统的每个用户设置环境信息,当用户第一次登录时,该文件被执行.并从/etc/profile.d目录的配置文件中搜集shell的设置.
>/etc/bashrc:  为每一个运行bash shell的用户执行此文件.当bash shell被打开时,该文件被读取.
>~/.bash_profile: 每个用户都可使用该文件输入专用于自己使用的shell信息,当用户登录时,该文件仅仅执行一次!默认情况下,他设置一些环境变量,执行用户的.bashrc文件.
>~/.bashrc: 该文件包含专用于你的bash shell的bash信息,当登录时以及每次打开新的shell时,该该文件被读取.
>~/.bash_logout: 当每次退出系统(退出bash shell)时,执行该文件. 
>
>
>另外,``/etc/profile``中设定的变量(全局)的可以作用于任何用户,而~/.bashrc等中设定的变量(局部)只能继承``/etc/profile``中的变量,他们是"父子"关系.

然后再运行一下.bashrc，这样本次session就可以使用这些环境变量了。

`source .bashrc`

同时也需要修改hadoop本身配置文件里的环境变量

```bash
export JAVA_HOME=/opt/jdk1.8.0_201
export HADOOP_CONF_DIR=${HADOOP_CONF_DIR:-"/home/hadoop/hadoop/hadoop-3.2.0/etc/hadoop"}
```

###以Pseudo-Distributed 模式运行

根据[Hadoop文档](http://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html#Configuration)，修改 安装目录/etc/hadoop/core-site.xml，如下：

```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
    <property>
		<name>hadoop.tmp.dir</name>
		<value>/home/hadoop/hadoop/hadooptmpdata</value>
    </property>
</configuration>
```

然后修改 安装目录/etc/hadoop/hdfs-sizte.xml，如下

```xml
<configuration>
    <property>
    	<name>dfs.replication</name>
    	<value>1</value>
    <property>
    	<name>dfs.name.dir</name>
    	<value>file:///home/hadoop/hdfs/namenode</value>
    </property>
    <property>
    	<name>dfs.data.dir</name>
    	<value>file:///home/hadoop/hdfs/datanode</value>
    </property>
</configuration>
```

这样是为了确定Hadoop启动的时候使用的hdfs实际文件位置，可以参考[hdfs手册](http://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/hdfs-default.xml)

为了运行MapReduce任务，需要配置YARN，同样的，修改 mapred-site.xml

```xml
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
```

yarn-site.html

```xml
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
</configuration>
```

### 测试HDFS

因为之前已经加入了环境变量，所以可以直接运行

`start-dfs.sh`

然后使用

`hdfs dfs -mkdir /test` 命令，就可以在hdfs根目录下创建了一个 test 文件夹。

然后 `hdfs dfs -ls /` ，可以看到

```
Found 1 items
drwxr-xr-x   - hadoop supergroup          0 2019-01-27 10:59 /test
```



