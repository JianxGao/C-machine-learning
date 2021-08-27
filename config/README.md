# 一、环境配置

本书中所有代码讲解与运行均以百度AI Studio作为开发平台。因此，下面我们以运行LSTM算法为例，为读者讲解如何使用百度Studio，创建机器学习项目并运行。

## 1.1 百度AI Studio简介

百度AI Studio是针对AI学习者的在线一体化学习与实训社区。平台集合了AI教程，深度学习样例工程，各领域的经典数据集，云端的超强运算及存储资源，以及比赛平台和社区。我们将依托于这一平台讲解与运行机器学习算法。

## 1.2 百度AI Studio配置

### 账号登录

我们在打开浏览器，输入网址：https://aistudio.baidu.com/aistudio/index，进入百度Studio，点击右上角【登录】，读者选择合适的登陆方式即可。

<img src="../content/image-20210523213039756.png" alt="image-20210523213039756" style="zoom:67%;" />

### 创建项目

登陆后，点击右上角的个人头像，点击【个人中心】进入个人主页，点击【项目】右侧的下拉三角，点击【创建和Fork的项目】，点击【创建项目】。

![image-20210523213205581](../content/image-20210523213205581.png)

我们选择类型为Notebook，点击【下一步】，

<img src="../content/image-20210523213249119.png" alt="image-20210523213249119" style="zoom:67%;" />

配置环境选择默认的 `PaddlePaddle 2.1.0`和 `python3.7`即可，点击【下一步】，根据实际情况填写项目描述，点击【创建】，完成项目的创建。

<img src="../content/image-20210523213325104.png" alt="image-20210523213325104" style="zoom:67%;" />

项目创建完成后，点击【启动环境】，我们在本例中使用免费的CPU基本版，点击【确定】，

<img src="../content/image-20210523213420514.png" alt="image-20210523213420514" style="zoom:67%;" />

等待片刻后，我们进入如下图所示的界面：

<img src="../content/image-20210523213536513.png" alt="image-20210523213536513" style="zoom:67%;" />

## 1.3 百度AI Studio案例运行

在该部分中，我们从github中将所有代码下载至AI Studio，并安装所有依赖，再通过随机森林算法演示如何在AI Studio运行机器学习代码。

### 下载代码

我们点击【终端】，输入指令：

```bash
git clone https://github.com/Jianx-Gao/C-machine-learning.git
```

![image-20210827153717909](..\content\image-20210827153717909.png)

如果下载失败，我们可以从github的镜像中下载代码：

```bash
git clone https://github.com.cnpmjs.org/Jianx-Gao/C-machine-learning.git
```

下载完成后，如下图所示：

![image-20210827154241403](..\content\image-20210827154241403.png)

### 环境配置

我们切换目录，进入`cd C-machine-learning`文件夹，同时点击左侧【C-machine-learning】文件夹

```bash
cd C-machine-learning
```

![image-20210827154313854](..\content\image-20210827154313854.png)

我们安装`requirements.txt`中写的所有算法的依赖

```bash
pip install -r requirements.txt
```

### 案列运行

下面，我们就可以运行我们的案例来测试代码了，我们分别测试C语言和Python版本的随机森林算法，检验环境是否安装成功

##### C语言版本

```bash
cd Random_Forest/C_version
bash compile.sh
```

![image-20210827155207533](..\content\image-20210827155207533.png)

##### Python语言版本

```bash
cd ../Python_version
python run.py
```

![image-20210827155433689](..\content\image-20210827155433689.png)