
import copy
import torch.nn as nn

from .resnet_cifar import resnet32
from .resnet import resnet10, resnet18, resnet50
from .hierNet import hiernet
from .pivot import pivot

def get_model(model_dict, nodes=None):

    name = model_dict['arch']
    feat_size = model_dict.get('feat_size', 512)
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')
    param_dict.pop('feat_size', None)

    if name == 'hiernet':
        param_dict['input_size'] = feat_size
        param_dict['nodes'] = nodes
        model = model(**param_dict)

    else:
        model = model(**param_dict)

    return model

def _get_model_instance(name):
    try:
        return {
            'resnet10': resnet10,
            'resnet18': resnet18,
            'resnet32': resnet32,
            'resnet50': resnet50,
            'hiernet': hiernet,
            'pivot': pivot
        }[name]
    except:
        raise ('Model {} not available'.format(name))
