from paper_1.evaluation.classification_metrics import ClassificationMetrics
from paper_1.utils import create_classification_labels
from typing import Union, Generator
from torch.optim import Optimizer
from torch.nn import Module
from copy import deepcopy
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import sys


def compute_discriminator_labels(data_src: list, data_tar: list) -> tuple:
    """"""

    # get the labeled data fom source domain
    num_src_data = len(data_src)

    # get the labeled or unlabeled data fom target domain
    num_tar_data = len(data_tar)

    # compute the discriminator labels for the source and target domains
    dis_src = torch.ones(num_src_data).long()
    dis_tar = torch.zeros(num_tar_data).long()
    return dis_src, dis_tar


def compute_adversarial_domain_loss(model: Module, discriminator: Module, data_src: list,
                                    data_tar: list, metric, gamma: float) -> Tensor:
    """"""

    # compute the labels for the discriminator
    dis_src, dis_tar = compute_discriminator_labels(data_src, data_tar)

    # compute the features for the source and the target domain
    features_src = model.encoder(data_src)
    features_tar = model.encoder(data_tar)

    # concatenate source and target domain features and labels
    domain_features = torch.cat([features_src, features_tar])
    domain_labels = torch.cat([dis_src, dis_tar])
    domain_labels = domain_labels.to(domain_features.device)

    # compute the predictions of the discriminator
    domain_predictions = discriminator(domain_features, gamma)

    # compute the loss
    domain_loss = metric.compute_metrics(domain_predictions, domain_labels, loss='cross_entropy', scalars=False, images=False)['loss']
    return domain_loss


def train_epoch(model: Module, discriminator: Module, optimizer: Optimizer, metric_obj: ClassificationMetrics,
                num_iter: int, metric_params: dict, train_loader_src: Generator, train_loader_tar: Generator,
                src_domain: str, epoch: int, num_epochs: int) -> dict:
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

            # compute the classification loss
            class_loss = metric_obj.compute_metrics(predictions_src, labels_src, loss='cross_entropy',
                                                    scalars=False, images=False, logits=True)['loss']

            # compute the gamma factor
            p = float(i + epoch * num_iter) / (num_epochs * num_iter)
            gamma = 1 * (2. / (1. + np.exp(-10 * p)) - 1)

            # compute the domain adversarial loss
            dann_loss = compute_adversarial_domain_loss(model, discriminator, data_src, data_tar, metric_obj, gamma)

            # compute the complete loss
            total_loss = class_loss
            total_loss += dann_loss

            # clear old gradients
            optimizer.zero_grad()

            # backpropagate the error
            total_loss.backward()

            # update the model
            optimizer.step()

            # store current predictions and labels
            prediction_list.append(predictions_src.detach())
            label_list.append(labels_src.detach())

            # compute the running average loss
            avg_epoch_loss = avg_epoch_loss * float(i) / (i + 1) + class_loss.detach().cpu().item() / (i + 1)
            del class_loss
            del dann_loss
            del total_loss

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
