之前在把自己用typora写的文章发布到知乎上面时，一直使用的是makedown插件，这款Chrome插件可以很方便地转换数学公式，因此在文章里有大量数学公式时显得尤为方便。

然而最近这款插件好像没用了，公式只能一个一个复制上去，实在是令人绝望。作为一个程序员，最反感的就是这种重复性工作，于是首先考虑能不能看看makedown插件哪里出了bug。

于是进入开发者模式，果然Console报错了，定位一下：

```javascript
function findReactComponent (elem) {
    for (let key in elem) {
        if (key.startsWith('__reactInternalInstance$')) {
            return elem[key]._currentElement._owner._instance
        }
    }
    return null
}
```

貌似是这个函数出了错，这个elem应该是html里面的div元素（'.PublishPanel-wrapper'），那这个key什么鬼？那个元素下面也就两个子元素，什么是“__reactInternalInstance$”？看得我一头雾水，只能放弃。

这时候换个思路，能否直接看浏览器发出的网络请求，可以得到每次保存草稿的时候，浏览器发送了一个名为“draft”的patch请求，其中请求的payload如下：

```json
{"content":"<h2>Header</h2><pre lang=\"\">code1<br/>code2</pre><p> <img eeimg=\"1\" src=\"//www.zhihu.com/equation?tex=a%5E2%20%2B%20b%5E2%20%3D%20c%5E2%0A\" alt=\"a^2 + b^2 = c^2 \"/> </p><p><b>Bold </b><i>Italic<br/></i>new line</p><p>new paragraph</p>","delta_time":17}
```

从中可以看出保存草稿时，实际上就是把整篇文章转换成html发送到了后台，也可以看出数学公式实际上转化成了一个img，src为知乎的一个GET API。

于是接下来的事情就简单了：

1. 把markdown文件处理成html格式
2. 把处理好的字符串用patch方法发送

于是写了一个简单的python脚本Draft.py，需要准备两个文件，分别是文章和请求的header（header可以用chrome的开发者工具抓包得到）。然后按照如下格式执行：

```powershell
./Draft.py url article.md header.txt
```

然后刷新浏览器，发现已经是自己上传的版本了，耶！