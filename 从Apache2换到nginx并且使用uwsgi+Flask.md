 

## 起因

一开始我在云服务器上使用Flask作为小程序的后台服务器，然后直接运行Python脚本（为了能够在后台运行，使用 screen），但是发现Flask并不稳定，运行一段时间后似乎会自动失去端口，然后出现无法访问的情况。为了解决这个问题，我打算采用 Nginx 作为后端服务器，然后用 Uwsgi 将其和 Flask 应用连接起来。

## 安装和配置Nginx

安装十分简单，一般来说只需要`sudo apt-get install nginx`就可以，注意在安装前需要先停止Apache2的服务，否则将会安装失败（可能是因为80端口被占用）。

Nginx的默认的配置文件在 /etc/nginx 下。启动方法是直接 `sudo nginx`，停止方法则是`sudo nginx -s stop`

## 把wordpress应用从Apache2迁移到Nginx

因为一开始的个人博客是用Apache2作为http服务器的，所以现在要迁移到Nginx上。首先需要安装 php-fpm，直接使用`sudo apt-get install php-fpm`即可安装。

php-fpm(FastCGI Process Manager) 是一个用于在服务器和php后端数据交互的接口，让服务器（Nginx）直到，对于 *.php 文件，首先要给php解释器执行，然后把结果传回服务器，再返回给请求者。CGI实现了这个接口，FastCGI以更快地方式实现，而php-fpm也就是一个fastCGI的php官方版本。

安装之后，php7.0-fpm成为系统服务，在/etc/init.d文件夹下。可以用`sudo service php7.0-fpm start`启动，这样php-fpm就开始监听了。php7.0-fpm的配置文件在/etc/php/7.0/fpm下。



 然后要修改Nginx的配置文件，在/etc/nginx目录下，首先要在nginx.conf的http项里加入

```
upstream php
{
    server unix:/run/php/php7.0-fpm.sock
}
```

然后在sites-available/default中加入

```
server {
	listen	80;
	server_name	xxx.com; //你的域名，如果有多个域名可以用空格分开
	root	/var/www/wordpress-1; //网站的根文件夹
	index	index.php; //首页地址
	location / {
		try_files $uri $uri/ /index.php?$args;
	}
	//处理php文件
	location ~ \.php$ {
        try_files $uri =404;
        include fastcgi_params;
        fastcgi_pass php;  
        //如果前面没有定义upstream，那么和  fastcgi_pass unix:/run/php/php7.0-fpm.sock是一样的
		fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;	
	}
}

```

这个upstream 其实是一个负载均衡的办法，详见官方文档[Using nginx as HTTP load balancer](http://nginx.org/en/docs/http/load_balancing.html)。

这个配置文件里 `localtion ~ \.php$`项的意思大概是，把php文件转发给一个地址，既可以是远程地址，也可以是本地地址，这里因为默认的 /etc/php/7.0/fpm/pool.d/www 里面设置了`listen = /run/php/php7.0-fpm.sock`，表明php文件应该转移到这个本地的socket文件里。

## Uwsgi配置

首先创建任意一个配置文件，比如uwsgi.ini

```
[uwsgi]
http=127.0.0.1:8888
wsgi-file=xxx(你的python文件，必须包含Flask)
callable=app
touch-reload=/home/ubuntu/tutorServer/
```

然后按照如下命令启动

`sudo uwsgi --ini uwsgi.ini`

就可以启动uwsgi了。

注意这里的uwsgi 可能是通过pip安装的。后来我把Flask部署到新的服务器时，发现如果直接apt-get install uwsgi, 还要安装`uwsgi-plugin-python3`，然后配置文件如下：

```
[uwsgi]
plugin=python3
master=true
http-socket=127.0.0.1:8888
python-path=/home/ubuntu/server/TutorServer/
wsgi-file=/home/ubuntu/server/TutorServer/ServerMain.py
callable=app
touch-reload=/home/ubuntu/server/TutorServer/
pidfile=uwsgi.pid
```



再在nginx.conf里面加入

```
server {
	listen	443;
	server_name	ustczf.com;
	（实现https）
	ssl on;
    ssl_certificate xxx.crt;
    ssl_certificate_key xxx.key;
	location /{
		proxy_pass	http://127.0.0.1:8888;
	}
}
```

重启 nginx 就可以。

**注意，更为推荐的方法是直接在Nginx的配置文件里进行设置完毕**

