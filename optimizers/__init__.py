
import logging
import copy
from torch.optim import SGD, Adam

from .scheduler import step_scheduler

logger = logging.getLogger('mylogger')

def get_optimizer(opt_dict):
    """Function to get the optimizer instance.
    """
    name = opt_dict['name']
    optimizer = _get_opt_instance(name)
    param_dict = copy.deepcopy(opt_dict)
    param_dict.pop('name')
    logger.info('Using {} optimizer'.format(name))

    return optimizer, param_dict

def _get_opt_instance(name):
    try:
        return {
            'sgd': SGD,
            'adam': Adam,
        }[name]
    except:
        raise ('Optimizer {} not available'.format(name))


