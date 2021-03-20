



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

**通过logging.basicConfig函数对日志的输出格式及方式做相关配置**

logging.basicConfig函数各参数:
 filename: 指定日志文件名
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

