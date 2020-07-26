
import os
import logging
import datetime
from PIL import Image
from collections import OrderedDict


def flist_reader(flist):
    """Function to read the list of image files.
    file list format: image_path,image_label,image_index (similar to caffe's filelist).
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imindex = line.strip().split(',')
            imlist.append((impath, int(imlabel), int(imindex)))
    return imlist


def default_loader(path):
    return Image.open(path).convert('RGB')


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def assign_learning_rate(optimizer, lr=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def add_weight_decay(params, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in params:
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def get_logger(logdir):
    """Function to build the logger.
    """
    logger = logging.getLogger('mylogger')
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def cvt2normal_state(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
    module state_dict inplace, i.e. removing "module" in the string.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

