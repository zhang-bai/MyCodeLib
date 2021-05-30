



jupyter 配置

# **配置服务器远端登录**

1）生成配置文件

2）生成密码（后续写配置文件、登录Jupyter notebook需要）

打开python终端

In [1]: from IPython.lib import passwd

In [2]: passwd()

Enter password:

Verify password:

Out[2]: 'sha1:一些数字字母'

3）修改默认配置文件

$vim ~/.jupyter/jupyter_notebook_config.py

```
$jupyter notebook --generate-config

c.NotebookApp.ip='*'

c.NotebookApp.password = 'sha1:xxxxx'

c.NotebookApp.open_browser = False

c.NotebookApp.port =11001 #随便指定一个端口

c.IPKernelApp.pylab = 'inline'
c.NotebookApp.token = ''
c.NotebookApp.notebook_dir = u'/home/zhangbai'		# 配置默认目录
```





**Jupyter Notebook运行指定的conda虚拟环境**

```python
conda install -n python_env ipykernel
```

首先安装ipykernel：conda install ipykernel

 激活conda环境： source activate 环境名称

在虚拟环境下创建kernel文件：conda install -n 环境名称 ipykernel

python -m ipykernel install --user --name 环境名称 --display-name "Python (环境名称)"



**启动Jupyter**

```
$ jupyter notebook --allow-root
```



# jupyter 快捷方式

依次点击开始-Anaconda3(64-bit)，在Jupyter Notebook (Anaconda3)上右击，选择更多-打开文件位置。

选中Jupyter Notebook (Anaconda3)这个快捷方式，Ctrl+C，Ctrl+V，也就是把这个快捷方式复制一份，右击生成Jupyter Notebook (Anaconda3) - 副本，选择属性

在快捷方式-目标中：

将[jupyter-notebook-script.py](https://link.zhihu.com/?target=http%3A//jupyter-notebook-script.py/) 替换为[jupyter-lab-script.py](https://link.zhihu.com/?target=http%3A//jupyter-lab-script.py/)

形成如：

D:\Anaconda3\python.exe d:\Anaconda3\[cwp.py](https://link.zhihu.com/?target=http%3A//cwp.py/) d:\Anaconda3 d:\Anaconda3\python.exe d:\Anaconda3\Scripts\[jupyter-lab-script.py](https://link.zhihu.com/?target=http%3A//jupyter-lab-script.py/) "D:\data"

的形式，D:\data为Jupyter lab启动后的工作目录，起始位置和此保持一致。或是将"D:\data"删除，只在起始位置设置。如果启动目录还是无法修改可以尝试找到jupyter_notebook_config.json文件，将里面的
{}

修改其内容为：

{

  "NotebookApp": {

​    "notebook_dir": "D:/data"

  }

}

在备注中也可以修改自己喜欢的名字。也可以更换为自己喜欢的图标。

![img](https://pic2.zhimg.com/80/v2-2f05cf1b650fd76f4d334c891f6c2795_720w.jpg)

在常规中将名字改为自己喜欢的名字

![img](https://pic1.zhimg.com/80/v2-bbf99d43f2c29f25139d926f5c5efda4_720w.jpg)

然后在开始菜单下Anaconda3(64-bit)中就会形成刚才的快捷方式。



# **tmux 开启后台运行**

```text
# 启动命名tmux
$ tmux new -s <name>
```

重接会话

我们通过`tmux detach`关闭tmux伪窗口后，希望能再次进入某一个会话窗口，怎么操做？

```text
# 重接会话 使用伪窗口编号
$ tmux attach -t 0

# 重接会话 使用伪窗口名称
$ tmux attach -t xiaoqi
```





SSH 登录

vi authorized_keys

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD0MYaQw2qgWiK8yePvg86sp0A0ig6tBlkeqPOLN8zIGurbyNR8UbonlZh4Dr+jyUkaAf9LTCyo82bCRJLmL5DYxFG3blvTaiWpzxujcAPffLHKLDHe/39NUmHFi5t9TNiDaWurZPeesh55Mq9dCCElDRESqlQe6XYZ/mb3+fpBgXOo+3AZo+boYlMBqDQ5IvfhzHQR2QAChJRjBYPtPXEvNBYGUKuism7R8NnuGM8kGiqUbKQzWdtKuE7ydjGIl4qndOHRbuzsSHtRcT+FkzZ1aIeeHbDDGUHYAECIi8QmMjLRRguLAOoOg8eA3V6xYhV6bkVp2peDsuFHJWocwPGf 
```



# 测试端口 tcping

在windows下，我们可以下载tcping这个小工具来帮助我们查看指定的端口是否是通的。

https://elifulkerson.com/projects/tcping.php  （下载地址）

进去后，直接下载tcping.exe 那个文件就行。然后把下载好的工具放到电脑的C盘>Windows>System32 下面就行。

然后我们直接重新打开CMD窗口，输入命令：tcping 指定的IP或者域名 端口号 。输入完回车就可以查看这个IP的端口是否是通着的。

比如：tcping 10.20.66.37 8090



# BAT CMD脚本

cmd /k "cd /d d:\Files&&jupyter lab"


# Logging

## 基本使用

### 1） 简单的将日志打印到屏幕



```
import logging  
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s') 
logging.debug('this is debug message')  
logging.info('this is info message')  
logging.warning('this is warning message')  

#打印结果：WARNING:root:this is warning message  
```

默认情况下，logging将日志打印到屏幕，日志级别为WARNING；
 日志级别大小关系为：`CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET` ，当然也可以自己定义日志级别。

```python
import logging
find_seed = './seed_acc.csv'
message = 'Seed' + ' Acc' + ' Best_acc' + '\n'
with open(find_seed, 'w') as f:
	f.write(message)
logging.basicConfig(filename=find_seed, filemode='a', level=logging.INFO, format='%(message)s')
for seeds in seed_list:
    acc, best_acc = train_main(args, seeds)
    acc = round(acc, 4)
    best_acc = round(best_acc, 4)

    mess = str(seeds) + ' ' + str(acc) + ' ' + str(best_acc)
    print(mess)
    logging.info(mess)
```



**通过logging.basicConfig函数对日志的输出格式及方式做相关配置**

logging.basicConfig函数各参数:
 **filename: 指定日志文件名**
 filemode: 和file函数意义相同，指定日志文件的打开模式，'w'或'a'
 format: 指定输出的格式和内容，format可以输出很多有用信息，如上例所示:
  %(levelno)s: 打印日志级别的数值
  %(levelname)s: 打印日志级别名称
  %(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
  %(filename)s: 打印当前执行程序名
  %(funcName)s: 打印日志的当前函数
  %(lineno)d: 打印日志的当前行号
  %(asctime)s: 打印日志的时间
  %(thread)d: 打印线程ID
  %(threadName)s: 打印线程名称
  %(process)d: 打印进程ID
  %(message)s: 打印日志信息
 datefmt: 指定时间格式，同time.strftime()
 level: 设置日志级别，默认为logging.WARNING
 stream: 指定将日志的输出流，可以指定输出到sys.stderr,sys.stdout或者文件，默认输出到sys.stderr，当stream和filename同时指定时，stream被忽略



# R

**安装R内核**

```
conda install -c r r-essentials
```

**安装rpy2----可同时运行Python、R**

 `pip install rpy2`或者`conda install rpy2`

#### python 调用R

python对象转换成R对象

      通常，可以将python的list对象，转换成为R的vector对象【robjects.ListVector()将python的字典（或list）转换成R的列表】，之后直接使用R函数调用。rpy2提供了几个函数，供我们把将python的list转化成R的不同数据类型的vector，对应的函数有 robjects.IntVector(),robjects.FloatVector()等，具体如下：详见：rpy2的vector相关的官方文档
    
    robjects.StrVector()#字符
    robjects.IntVector()#整数
    robjects.FloatVector()#浮点
    robjects.complexVector()#复数
    robjects.FactorVector()#因子
    robjects.BoolVector()#布尔向量
    robjects.ListVector()#列表

需注意：使用vector系列函数时，输入的只能是python的列表，而不能是数字或者字符串。



# Tensorboard



tensorboard --logdir=<your_log_dir>

其中的 <your_log_dir> 既可以是单个 run 的路径，如上面 writer1 生成的 runs/exp；也可以是多个 run 的父目录，如 runs/ 下面可能会有很多的子文件夹，每个文件夹都代表了一次实验，我们令 --logdir=runs/ 就可以在 tensorboard 可视化界面中方便地横向比较 runs/ 下不同次实验所得数据的差异。


#### SummaryWriter()

参数为：def __init__(self, log_dir=None, comment='', **kwargs): 其中log_dir为生成的文件所放的目录，comment为文件名称。默认目录为生成runs文件夹目录

#### writer.add_scalar()

第一个参数可以简单理解为保存图的名称，第二个参数是可以理解为Y轴数据，第三个参数可以理解为X轴数据。当Y轴数据不止一个时，可以使用writer.add_scalars()

最后调用writer.close()









