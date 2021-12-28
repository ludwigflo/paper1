from typing import Union
import shutil
import torch
import yaml
import os


def read_parameter_file(parameter_file_path: str) -> dict:
    """
    Reads the parameters from a yaml file into a dictionary.

    Parameters
    ----------
    parameter_file_path: Path to a parameter file.

    Returns
    -------
    params: Dictionary containing the parameters defined in the provided yam file
    """

    with open(parameter_file_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def initialize_experiment_folder(base_path: str, source_domain: str, target_domain: str, val_fold: int) -> str:
    """
    Creates a Folder for the current experiment, which is defined by the training domain, training fold and the
    number of data points, ued for training the model.

    Parameters
    ----------
    base_path: Path, in which the new path is created.
    source_domain: Name of the source domain.
    target_domain: Name of the target domain.
    val_fold: Validation fold for the current experiment.
    """

    # compute the path, which should be created
    folder_name = base_path + source_domain + '/' + target_domain + '/' + str(val_fold) + '/'

    # create the path, if it doesn't already exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # create the path, if it doesn't already exist
    if not os.path.exists(folder_name + 'parameters/'):
        os.makedirs(folder_name + 'parameters/')
    return folder_name


def create_experiment_directory(base_path: str, source_domain: str, target_domain: str, val_fold: int,
                                file_list: list, experiment_id: int, quantile: Union[None, float] = None) -> str:
    """"""

    # extend the base_path
    base_path += str(experiment_id) + '/'
    if quantile is not None:
        base_path += str(quantile) + '/'

    # create a new directory, in which the current experiment is stored.
    exp_dir = initialize_experiment_folder(base_path, source_domain, target_domain, val_fold)

    target_folder = exp_dir + 'parameters/'
    for file in file_list:
        shutil.copyfile(file, target_folder + file.split('/')[-1])
    return exp_dir


# noinspection PyArgumentList
def create_classification_labels(labels: list, mapping_dict: dict) -> torch.LongTensor:
    """
    Creates a Pytorch Tensor containing classification labels based on a list of provided label names.

    Parameters
    ----------
    labels: List of label names.
    mapping_dict: Dictionary which maps labels to their indices.

    Returns
    -------
    labels: Label tensor, stored at the required device.
    """

    # convert label names to label indices
    label_indices = [mapping_dict[label] for label in labels]

    # convert the labels into a pytorch tensor and store it at the provided device
    labels = torch.LongTensor(label_indices)
    return labels
