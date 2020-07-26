
import copy
from loader.hier_dataset import hierdata, hierdata_cifar

def get_hierdataset(data_dict, batch_size, n_workers, splits):
    """Function to build data loader(s) for the specified splits given the parameters.
    """
    param_dict = copy.deepcopy(data_dict)
    param_dict['batch_size'] = batch_size
    param_dict['num_workers'] = n_workers

    return hierdata(splits, **param_dict)

def get_hierdataset_cifar(data_dict, batch_size, n_workers, splits):
    """Function to build data loader(s) for the specified splits given the parameters.
    """
    param_dict = copy.deepcopy(data_dict)
    param_dict['batch_size'] = batch_size
    param_dict['num_workers'] = n_workers

    return hierdata_cifar(splits, **param_dict)

