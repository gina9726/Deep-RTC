# Deep-RTC [[project page]](http://www.svcl.ucsd.edu/projects/deep-rtc)
This repository contains the source code accompanying our ECCV 2020 paper.  

[<b>Solving Long-tailed Recognition with Deep Realistic Taxonomic Classifier</b>](https://arxiv.org/abs/2007.09898)  
[Tz-Ying Wu](http://www.svcl.ucsd.edu/people/gina), [Pedro Morgado](http://www.svcl.ucsd.edu/~morgado), [Pei Wang](http://www.svcl.ucsd.edu/~peiwang), [Chih-Hui Ho](http://www.svcl.ucsd.edu/people/johnho), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno)
```
@inproceedings{Wu20DeepRTC,
	title={Solving Long-tailed Recognition with Deep Realistic Taxonomic Classifier},
	author={Tz-Ying Wu and Pedro Morgado and Pei Wang and Chih-Hui Ho and Nuno Vasconcelos},
	booktitle={European Conference on Computer Vision (ECCV)},
	year={2020}
}
```
# Dependencies
- Python (3.5.6)
- PyTorch (1.2.0)
- torchvision (0.4.0)
- NumPy (1.15.2)
- Pillow (5.2.0)
- PyYaml (5.1.2)
- tensorboardX (1.8)

# Data preparation
- CIFAR100 
[[Raw images]](https://www.cs.toronto.edu/~kriz/cifar.html)
[[Long-tail version]](https://github.com/richardaecn/class-balanced-loss)
- AWA2
[[Raw images]](https://cvml.ist.ac.at/AwA2/)
- ImageNet
[[Raw images]](http://image-net.org/download-images)
[[Long-tail version]](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)
- iNaturalist
[[Raw images]](https://github.com/visipedia/inat_comp/blob/master/2018/README.md)  

These datasets can be downloaded from the above links.
Please organize the images in the hierarchical folders that represent the dataset hierarchy, and put the root folder under `prepro/raw`. For example,
```
prepro/raw/imagenet
--abstraction
----bubble
------ILSVRC2012_val_00014026.JPEG
------ILSVRC2012_val_00000697.JPEG
...
--physical_entity
----object
...
```
While CIFAR100 and iNaturalist have released taxonomies, we built the tree-type taxonomy of AWA2 and ImageNet with [WordNet](https://wordnet.princeton.edu/). All the taxonomies are provided in `prepro/data/{dataset}/tree.npy`, and the data splits are provided in `prepro/splits/{dataset}/{split}.json`. Please refer to `prepro/README.md` for more details. After the raw images are managed hierarchically, run
```
$ ./prepare_data.sh {dataset}
```
where {dataset}=awa2/cifar100/imagenet/inaturalist. This will automatically generate the data lists for all splits, and build the codeword matrices needed for training Deep-RTC. Note that our codes can be applied to other datasets once they are organized hierarchically.  

# Training and evaluation
To train and evaluate Deep-RTC, run
```
$ export PYTHONPATH=${PWD}/prepro:${PYTHONPATH}
$ ./run.sh {dataset}
```
where {dataset}=awa2/cifar100/imagenet/inaturalist.
Our pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1BNoED2LlER-TDw1bMXG8IGZ858zlTobO?usp=sharing).


