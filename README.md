# model
The CNN model base on humanseg

[![Build Status](https://travis-ci.com/paddleseg/model.svg?branch=master)](https://travis-ci.com/paddleseg/model)

## 模型描述

模型基于[PaddlePaddle](https://www.paddlepaddle.org.cn/)提供的预训练模型[PaddleSeg](https://www.paddlepaddle.org.cn/modelbasedetail/deeplabv3plus)进行Fine Tune而得到。

使用的是supervisely数据集。

### 模型指标

acc=0.9818 IoU=0.9536

## 如何运行

建议采用docker来运行此服务。

```shell
docker run -it --rm --name app -p 80:80 registry.cn-zhangjiakou.aliyuncs.com/vikings/paddle:{version}
```

## 镜像列表

国内阿里源

* registry.cn-zhangjiakou.aliyuncs.com/vikings/paddle

* [Docker Hub](https://hub.docker.com/repository/docker/vikings/paddle)