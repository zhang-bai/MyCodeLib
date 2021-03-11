



jupyter 配置

**配置服务器远端登录**

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









**tmux 开启后台运行**

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