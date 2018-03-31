# Yellowbrick

[![Build Status](https://travis-ci.org/DistrictDataLabs/yellowbrick.svg?branch=master)](https://travis-ci.org/DistrictDataLabs/yellowbrick)
[![Coverage Status](https://coveralls.io/repos/github/DistrictDataLabs/yellowbrick/badge.svg?branch=master)](https://coveralls.io/github/DistrictDataLabs/yellowbrick?branch=master)
[![Code Health](https://landscape.io/github/DistrictDataLabs/yellowbrick/master/landscape.svg?style=flat)](https://landscape.io/github/DistrictDataLabs/yellowbrick/master)
[![Documentation Status](https://readthedocs.org/projects/yellowbrick/badge/?version=latest)](http://yellowbrick.readthedocs.io/en/latest/?badge=latest)
[![Stories in Ready](https://badge.waffle.io/DistrictDataLabs/yellowbrick.png?label=ready&title=Ready)](https://waffle.io/DistrictDataLabs/yellowbrick)


**帮助选择机器学习模型的可视化分析和诊断工具**

![沿着黄砖路走下去](docs/images/yellowbrickroad.jpg)

摄影 [Quatro Cinco](https://flic.kr/p/2Yj9mj), 经许可使用, Flickr Creative Commons.

这个README是针对开发者的指南，如果你对Yellowbrick感兴趣，请查看我们的[文档](http://www.scikit-yb.org/).

## 什么是Yellowbrick?

Yellowbrick是由一套被称为"Visualizers"组成的可视化诊断工具组成的套餐，其由Scikit-Learn API延伸而来，对模型选择过程起指导作用。总之，Yellowbrick结合了Scikit-Learn和Matplotlib并且最好得传承了Scikit-Learn文档，对 _你的_ 模型进行可视化！

![Visualizers](docs/images/visualizers.png)

### Visualizers

Visualizers也是estimators（从数据中习得的对象），其主要任务是产生可对模型选择过程有更深入了解的视图。从Scikit-Learn来看，当可视化数据空间或者封装一个模型estimator时，其和转换器（transformers）相似，就像"ModelCV" (比如 RidgeCV, LassoCV)的工作原理一样。Yellowbrick的主要目标是创建一个和Scikit-Learn类似的有意义的API。其中最受欢迎的visualizers包括：

#### 特征可视化

- **Rank Features**: 对单个或者两两对应的特征进行排序以检测其相关性
- **Parallel Coordinates**: 对实例进行水平视图
- **Radial Visualization**: 在一个圆形视图中将实例分隔开
- **PCA Projection**: 通过主成分将实例投射
- **Feature Importances**: 基于它们在模型中的表现对特征进行排序
- **Scatter and Joint Plots**: 用选择的特征对其进行可视化

#### 分类可视化

- **Class Balance**: 看类的分布怎样影响模型
- **Classification Report**: 用视图的方式呈现精确率，召回率和F1值
- **ROC/AUC Curves**: 特征曲线和ROC曲线子下的面积
- **Confusion Matrices**: 对分类决定进行视图描述

#### 回归可视化

- **Prediction Error Plots**: 沿着目标区域对模型进行细分
- **Residuals Plot**: 显示训练数据和测试数据中残差的差异
- **Alpha Selection**: 显示不同alpha值选择对正则化的影响

#### 聚类可视化

- **K-Elbow Plot**: 用肘部法则或者其他指标选择k值
- **Silhouette Plot**: 通过对轮廓系数值进行视图来选择k值

#### 文本可视化

- **Term Frequency**: 对词项在语料库中的分布频率进行可视化
- **TSNE**: 用随机邻域嵌入来投射文档

以及更多！Visualizers随时在增加中，请务必查看示例（甚至是develop分支上的），并且随时欢迎你对Visualizers贡献自己的想法。

## 安装Yellowbrick

Yellowbrick和Python 2.7及以后版本兼容，但是倾向于使用Python 3.5及以后版本，这样可发挥其全部功能。Yellowbrick依赖于Scikit-Learn 0.18及以后版本和Matplotlib 1.5及以后版本。安装Yellowbrick最简单的方法就是用来自于PyPI的pip方法 —— Python首选的包安装器。

    $ pip install yellowbrick

需要注意的是Yellowbrick是一个在建的项目，目前常规发布新的版本，并且每一个新版本都将会有新的可视化功能更新。为了将Yellowbrick升级到最新版本，你可以用如下pip命令.

    $ pip install -u yellowbrick

你也可以用 `-u` 标记对Scikit-Learn，matplotlib或者其他和Yellowbrick兼容的第三方包进行升级.

如果你使用的是Windows或者Anaconda，你也可以充分利用conda:

    conda install -c districtdatalabs yellowbrick

然而需要注意的是，在Linux上用Anaconda安装matplotlib时有一个 [已知的漏洞](https://github.com/DistrictDataLabs/yellowbrick/issues/205) 。

## 使用Yellowbrick

为了更好得配合Scikit-Learn一起使用，我们特意对Yellowbrick API进行了一些特殊设计。如下是一个比较典型的使用Scikit-Learn和Yellowbrick的工作流程。
The Yellowbrick API is specifically designed to play nicely with Scikit-Learn. Here is an example of a typical workflow sequence with Scikit-Learn and Yellowbrick:

### 特征可视化

下面这个例子将向我们展示Rank2D是怎样用一种特殊的算法对数据中的特征进行两两比较，然后将其结果排列至左下角三角区的视图。

```python
from yellowbrick.features import Rank2D

visualizer = Rank2D(features=features, algorithm='covariance')
visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data
```

### 模型可视化

这个例子里，我们将实例化一个Scikit-Learn分类器，然后我们用Yellowbrick的ROCAUC类来对分类中的敏感度和特异性之间的权衡进行可视化。

```python
from sklearn.svm import LinearSVC
from yellowbrick.classifier import ROCAUC

model = LinearSVC()
model.fit(X,y)
visualizer = ROCAUC(model)
visualizer.score(X,y)
visualizer.poof()
```

想要对如何开始使用Yellowbrick有更多了解，请查看我们的 [示例笔记](https://github.com/DistrictDataLabs/yellowbrick/blob/develop/examples/examples.ipynb).

还有 [快速开始指南](https://github.com/DistrictDataLabs/yellowbrick/blob/master/docs/quickstart.rst).

## 对Yellowbrick做出贡献

Yellowbrick是一个开源项目，其支持社区将会谦卑地接受你对项目所做的任何贡献，并对此表示感激。不论是大还是小，任何贡献都能使其发生大的变化；如果你从来没有对开源项目进行过贡献，我们希望你能从Yellowbrick开始！

主要来讲，Yellowbrick的开发是添加和创建 *visualizers* &mdash; 对象，其通过对数据的学习创造一个对数据或者模型的可视化呈现。针对某些特定目标，Visualizers还可与Scikit-Learn的estimators，transformers以及pipelines结合使用，使其更易于创建和部署。最常见的贡献方式就是为一个特定的模型或者模型家族创建一个visualizer。我们后面将会详细讨论怎样创建一个visualizer。

除了创建visualizer，还有很多做贡献的方法：

- 在 [GitHub Issues](https://github.com/DistrictDataLabs/yellowbrick/issues) 上提交漏洞报告或者特征请求。
- 在我们的示例库中用Jupyter notebook的方式贡献示例 [ gallery](https://github.com/DistrictDataLabs/yellowbrick/tree/develop/examples) 。
- 在 [user testing](http://www.scikit-yb.org/en/latest/evaluation.html) 上帮助我们
- 在我们网站上帮我们加入文档 [scikit-yb.org](http://www.scikit-yb.org) 。
- 对我们项目进行单元测试或者集成测试。
- 在issues，mailing list，Stack Overflow或者其他地方帮我们回答问题。
- 帮我们把文档翻译成其他语言。
- 写一个博客，tweet或者和他人分享我们的项目。
- 教别人使用Yellowbrick。

正如你看到的，你可以通过多种方式加入到我们当中来。并且我们也非常开心你能加入我们！我们唯一的要求是你遵守开放的原创，尊敬并且体谅他人，就如下面描述的一样 [Python Software Foundation Code of Conduct](https://www.python.org/psf/codeofconduct/) 。

想要了解更多，请查看  `CONTRIBUTING.md` 文件中repository的根目录或者详细的文档 [Contributing to Yellowbrick](http://www.scikit-yb.org/en/latest/contributing.html) 。

## 开发脚本

Yellowbrick包含对开发有帮助的脚本，包括下载测试数据的脚本和管理比较图片的脚本。

### 图片

图片比较帮手脚本通过将文件从 `actual_images` 文件夹拷贝到设置基线对测试目录的 `baseline_images` 文件夹进行管理。要用这个脚本的话，首先运行测试（会产生“image not found”错误产生），然后按照如下方式将图片拷入基线：

```
$ python -m tests.images tests/test_visualizer.py
```

`tests/test_visualizer.py` 是包含有图片比较测试的测试文件。所以相关的测试都会被发掘、证实，并且拷贝到基线目录。如果想要将图片从现行的或这基线的状态重设到测试状态，可用如下 `-C` 参数：

```
$ python -m tests.images -C tests/test_visualizer.py
```
可以用Glob语法来移除多个文件。比如，将所有分类器测试重设：

```
$ python -m tests.images tests/test_classifier/*   
```

然而推荐的的方法是有针对性进行测试，而不是对整个目录进行更新。

翻译：[Juan L. Kehoe](https://juan0001.github.io/)

