# 二值分割
趁电脑空闲，做了个显著性检测的baseline。试一下vgg, resnet系列和densenet系列的效果。

用DUTS-train训练，在ECSSD测试的效果：

|                   | FM   | MEA  |
| ----------        | -----| ---  |
| VGG               | .827 | .075 |
| DenseNet121       | .851 | .060 |
| DenseNet161       | .853 | .058 |
| DenseNet169       | .843 | .060 |
| DenseNet201       | .868 | .051 |
| ResNet50          | .821 | .069 |
| ResNet101         | .847 | .059 |
| ResNet152         | .856 | .056 |

pytorch: '0.5.0a0+9db779f'

## Usage
新的又做不动，也懒得复习，幻灯片又懒得做，闲着没事更新一下这个仓库。

### 训练：
```shell
python train.py 
--train_dir 'path/to/training/data' 
--val_dir 'path/to/validation/data' 
--q 类型 --b batchsize 
--check_dir 'path/to/save/parameters'
```
path/to/training/data里有两个文件夹名字叫images和masks，分别是训练图片(jpg)和真值(png)。

path/to/training/data里有两个文件夹名字叫images和masks，分别是验证图片和真值。
（我从训练图片里分了1000个作为validation）

类型：
'vgg', 
'resnet101', 
'resnet152', 
'resnet50', 
'resnet34', 
'resnet18', 
'densenet121', 
'densenet161', 
'densenet169', 
'densenet201'

训练得到的模型参数保存在path/to/save/parameters。只保存在validation数据上loss最小的模型。

不知为何，用adam训练resnet的效果不好。所以resnet用SGD训练，其余都用Adam。

### 测试：
```shell
python test.py
--input_dir 'path/to/test/data' 
--output_dir 'path/to/save/results' 
--para_dir 'path/to/trained/parameters' 
--q 类型
--b batchsize
```
path/to/test/data里有个文件夹叫images，里面是.jpg图片

path/to/save/results：保存产生结果的路径

path/to/trained/parameters：训练时保存模型参数的路径

类型：要和训练时相同。

评价效果：
修改evafunc.py里的
```python
base_dir = '/home/zeng/data/datasets/saliency_Dataset'
algs = ['duts_res50', 'duts_res101']
datasets = ['ECSSD', 'SOD']
```
比如base_dir里有ECSSD和SOD两个文件夹，这两个文件夹里有images, masks, duts_res50, duts_res101这些文件夹
，里面分别是图片(jpg)、真值(png)、两种方法产生的结果图(png)

然后运行它
