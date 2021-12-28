from copy import deepcopy
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import torch
import sys


def compute_pseudo_labels(model: nn.Module, data_tar: pd.DataFrame) -> pd.DataFrame:
    """"""

    with torch.no_grad():
        model.eval()
        probabilities = []
        predicted_classes = []
        print('Computing Pseudo Labels...')
        with tqdm(total=len(data_tar), file=sys.stdout) as pbar:
            for sample in data_tar['text_data'].tolist():
                predictions = model([sample], logits=False)[0, ...]

                predicted_class = torch.argmax(predictions, dim=0)
                predicted_classes.append(predicted_class.detach().cpu().item())
                probabilities.append(predictions.detach().cpu())
                pbar.update(1)
    data_tar_out = deepcopy(data_tar)
    data_tar_out['probabilities'] = probabilities
    data_tar_out['predicted_classes'] = predicted_classes
    return data_tar_out


def split_pseudo_labels(data_tar: pd.DataFrame, inverse_class_mapping: dict):
    """"""

    # get the indices of the classes
    class_indices = inverse_class_mapping.keys()

    # initialize a dict, which maps classes to data points
    split_dict = dict()

    for class_index in class_indices:

        # select all data points, which were classified as the current class
        sub_frame = data_tar.loc[data_tar['predicted_classes'] == class_index]

        # store the sub frame in the split dicitonary
        split_dict[inverse_class_mapping[class_index]] = sub_frame
    return split_dict


def select_quantile(data_frame: pd.DataFrame, quantile: float, class_mapping: dict) -> tuple:
    """"""

    predicted_classes = data_frame['predicted_classes'].tolist()
    probabilities = data_frame['probabilities'].tolist()
    probabilities = torch.Tensor([p[c].item() for p, c in zip(probabilities, predicted_classes)])
    data_frame['probabilities'] = probabilities

    # compute the quantile based on the certainty of the model
    q = torch.quantile(probabilities, 1-quantile).item()

    # select those data points, for which the predicted certainty is higher than the quantile value
    data_frame = data_frame.loc[data_frame['probabilities'] >= q]

    # convert class indices into class labels
    true_cls = data_frame['labels'].tolist()
    data_frame['labels'] = [class_mapping[label] for label in data_frame['predicted_classes'].tolist()]

    data_frame.drop(axis=1, labels=['probabilities', 'predicted_classes'], inplace=True)
    return data_frame


def accurracy(pseudo_labels: list, true_labels: list):

    acc = 0.0
    for pred, tar in zip(pseudo_labels, true_labels):
        if pred == tar:
            acc += 1
    acc /= len(pseudo_labels)
    return acc


def select_splitted_pseudo_labels(model: nn.Module, data_tar: list, quantile: float, class_mapping: dict):
    """"""

    # compute probabilities
    data_tar = compute_pseudo_labels(model, data_tar)

    # splt data according to the predicted classes
    split_dict = split_pseudo_labels(data_tar, class_mapping)

    # select the most certain samples in the provided quantile (for each class independently)
    data_out = []
    for key, data_frame in split_dict.items():
        selected_data = select_quantile(data_frame, quantile, class_mapping)
        data_out.append(selected_data)
    data_out = pd.concat(data_out)
    return data_out