from paper_1.evaluation.classification_metrics import ClassificationMetrics
from paper_1.utils import create_classification_labels
from torch.distributions.beta import Beta
from typing import Union, Generator
from torch.optim import Optimizer
from torch.nn import Module
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import sys


def labels_one_hot(labels: torch.Tensor) -> torch.Tensor:
    """"""

    num_classes = 3
    labels_one_hot = torch.zeros(labels.size()[0], num_classes)
    for i in range(labels.size()[0]):
        labels_one_hot[labels[i]] = 1
    labels_one_hot = labels_one_hot.to(labels.device)
    return labels_one_hot


def linear_combine_tensors(tensor_1: torch.Tensor, tensor_2: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """"""

    # check if the tensor dimensions match
    assert tensor_1.size() == tensor_2.size(), "Tensor size needs to be equal..."

    # compute the number of dimensions
    tensor_size = tuple(list(tensor_1.size())[1:])
    num_dimensions = len(tensor_1.size())
    weight_base_shape = [-1]
    for i in range(num_dimensions-1):
        weight_base_shape.append(1)

    # reshape the weights to the desired shape and compute the complementary tensor
    weights = weights.view(*tuple(weight_base_shape))
    comp_weights = 1 - weights
    weight_tensor = weights.repeat(1, *tensor_size)
    comp_weights_tensor = comp_weights.repeat(1, *tensor_size)

    # move the weight tensors to the correct device
    weight_tensor = weight_tensor.to(tensor_1.device)
    comp_weights_tensor = comp_weights_tensor.to(tensor_1.device)

    # compute the linear combination of the tensors
    mixup_tensor = weight_tensor * tensor_1 + comp_weights_tensor * tensor_2
    return mixup_tensor


def prepare_mixup_loss(features: torch.Tensor, targets: torch.Tensor, alpha: float = 2.0):
    """"""

    # shuffle the features and targets to create mixup samples
    idx = torch.randperm(features.size()[0])
    features_perm = features[idx].view(features.size())
    targets_perm = targets[idx].view(targets.size())

    # define the probability distribution
    distribution = Beta(alpha, alpha)

    # compute the weights (called lambda in the original approach) and complementary weights
    weights = distribution.rsample((features.size()[0],))

    # compute the linear combination between the features and the permutated features
    mixup_features = linear_combine_tensors(features, features_perm, weights)

    # compute the linear combination between the targets and the permutated targets
    mixup_targets = linear_combine_tensors(targets, targets_perm, weights)
    return mixup_features, mixup_targets


def compute_source_mixup_loss(features_src: torch.Tensor, labels: torch.Tensor, model: nn.Module, metric) -> torch.Tensor:
    """"""

    # compute one hot labels
    labels_src = labels_one_hot(labels)

    # compute mixup features and targets
    mixup_features_src, mixup_targets_src = prepare_mixup_loss(features_src, labels_src)

    # compute predictions based on the mixup features
    mixup_predictions_src = model.output_layer(mixup_features_src)

    # compute the mixup loss
    mixup_loss_src = metric.compute_metrics(mixup_predictions_src, mixup_targets_src, scalars=False,
                                            loss='soft_cross_entropy', images=False, logits=True)['loss']
    return mixup_loss_src


def compute_target_mixup_loss(data_tar: torch.Tensor, model: nn.Module, metric) -> torch.Tensor:
    """"""

    # compute features and soft pseudo labels on the target domain
    features_tar = model.encoder(data_tar)
    pseudo_labels_tar = model.output_layer(features_tar).detach()

    # compute mixup features and mixup labels on the target domain
    mixup_features_tar, mixup_targets_tar = prepare_mixup_loss(features_tar, pseudo_labels_tar)

    # compute the models predictions on the mixup features
    mixup_predictions_tar = model.output_layer(mixup_features_tar)

    # compute the mixup loss for the target domain
    mixup_loss_tar = metric.compute_metrics(mixup_predictions_tar, mixup_targets_tar, scalars=False,
                                            loss='l1', images=False, logits=True)['loss']
    return mixup_loss_tar


def train_epoch(model: Module, optimizer: Optimizer, metric_obj: ClassificationMetrics, num_iter: int, metric_params: dict,
                train_loader_src: Generator, train_loader_tar: Generator, src_domain: str) -> dict:
    """"""

    with tqdm(total=num_iter, file=sys.stdout) as pbar:

        # change to the train mode
        model.train()

        # initialization of required variables
        prediction_list = []
        avg_epoch_loss = 0
        label_list = []

        for i, (data_batch_src, done) in enumerate(train_loader_src):

            # prepare data and labels for the source domain
            data_src, labels_src = data_batch_src[0], data_batch_src[1]
            labels_src = create_classification_labels(labels_src, metric_params['class_mapping'])
            labels_src = labels_src.to(next(model.parameters()).device)

            # prepare the unlabeled data points for the target domain
            data_batch_tar, _ = next(train_loader_tar)
            data_tar = data_batch_tar[0]

            # compute predictions and store them as well as their corresponding labels
            features_src = model.encoder(data_src)
            predictions_src = model.output_layer(features_src)

            # store current predictions and labels
            prediction_list.append(predictions_src)
            label_list.append(labels_src)

            # compute the classification loss
            class_loss = metric_obj.compute_metrics(predictions_src, labels_src, loss='cross_entropy',
                                                    scalars=False, images=False, logits=True)['loss']

            mixup_loss_src = compute_source_mixup_loss(features_src, labels_src, model, metric_obj)
            mixup_loss_tar = compute_target_mixup_loss(data_tar, model, metric_obj)
            total_loss = class_loss + 0.1 * (mixup_loss_src + mixup_loss_tar)

            # clear old gradients
            optimizer.zero_grad()

            # backpropagate the error
            total_loss.backward()

            # update the model
            optimizer.step()

            # compute the running average loss
            avg_epoch_loss = avg_epoch_loss * float(i) / (i + 1) + class_loss.detach().cpu().item() / (i + 1)

            # print the progress
            if i % 10 == 9:
                pbar.set_description('Average training loss: ' '{:10.5f}'.format(avg_epoch_loss))
                pbar.update(10)

            if i == num_iter:
                break

        with torch.no_grad():

            # convert predictions and targets into the correct shape
            predictions = torch.cat(prediction_list).detach()
            labels = torch.cat(label_list).detach()

            # compute the metrics
            metrics_tmp = metric_obj.compute_metrics(predictions, labels, loss=None, scalars=True, images=True, logits=True)
            metrics = {}
            for key, values in metrics_tmp.items():
                if key not in metrics:
                    metrics[key] = {}
                for sub_key, sub_values in values.items():
                    metrics[key][sub_key + '/' + src_domain + '/training'] = sub_values
        return metrics


def validation_epoch(model: nn.Module, metric, num_val_iter_list: list, metric_params: dict,
                     data_loader_list: list, domain_name_list: list) -> dict:
    """"""

    with torch.no_grad():

        # switch to validation mode
        model.eval()

        # initialize necessary variables
        prediction_list = []
        avg_epoch_loss = 0
        label_list = []
        metrics = {}

        for j, data_loader in enumerate(data_loader_list):
            print()
            num_iter = num_val_iter_list[j]
            with tqdm(total=num_iter, file=sys.stdout) as pbar:

                for i, (data_batch, done) in enumerate(data_loader):

                    # prepare data and labels
                    data, labels = data_batch[0], data_batch[1]
                    labels = create_classification_labels(labels, metric_params['class_mapping'])
                    device = next(model.parameters()).device
                    labels = labels.to(device)

                    # compute predictions and store them as well as their corresponding labels
                    predictions = model(data, logits=True)
                    prediction_list.append(predictions)
                    label_list.append(labels)

                    # compute all required metrics and specify the loss value, which should be optimized
                    output = metric.compute_metrics(predictions, labels, loss='cross_entropy', scalars=False, images=False)
                    loss = output['loss']

                    # compute the running average loss
                    avg_epoch_loss = avg_epoch_loss * float(i) / (i + 1) + loss.detach().cpu().item() / (i + 1)

                    # print the progress
                    if i % 10 == 9:
                        pbar.set_description('Average validation loss {}: {:10.5f}'.format(domain_name_list[j],
                                                                                           avg_epoch_loss))
                        pbar.update(10)

                    if done:
                        pbar.update((i % 10) + 1)
                        break

                # convert predictions and targets into the correct shape
                predictions = torch.cat(prediction_list).detach()
                labels = torch.cat(label_list).detach()

            # compute the metrics
            metrics_tmp = metric.compute_metrics(predictions, labels, loss=None, scalars=True, images=True, logits=True)
            for key, values in metrics_tmp.items():
                if key not in metrics:
                    metrics[key] = {}
                for sub_key, sub_values in values.items():
                    metrics[key][sub_key + '/' + domain_name_list[j] + '/validation'] = sub_values
        return metrics
