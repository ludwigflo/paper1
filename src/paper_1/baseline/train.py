from paper_1.evaluation.classification_metrics import ClassificationMetrics
from paper_1.utils import create_classification_labels
from typing import Union, Generator
from torch.optim import Optimizer
from torch.nn import Module
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import sys


def train_epoch(model: Module, optimizer: Optimizer, metric_obj: ClassificationMetrics, num_iter: int,
                metric_params: dict, train_loader: Generator, src_domain: str) -> dict:
    """"""

    with tqdm(total=num_iter, file=sys.stdout) as pbar:

        # change to the train mode
        model.train()

        # initialization of required variables
        prediction_list = []
        avg_epoch_loss = 0
        label_list = []

        for i, (data_batch_src, done) in enumerate(train_loader):

            # prepare data and labels for the source domain
            data_src, labels_src = data_batch_src[0], data_batch_src[1]
            labels_src = create_classification_labels(labels_src, metric_params['class_mapping'])
            labels_src = labels_src.to(next(model.parameters()).device)

            # compute predictions and store them as well as their corresponding labels
            features_src = model.encoder(data_src)
            predictions_src = model.output_layer(features_src)

            # store current predictions and labels
            prediction_list.append(predictions_src)
            label_list.append(labels_src)

            # compute the classification loss
            class_loss = metric_obj.compute_metrics(predictions_src, labels_src, loss='cross_entropy',
                                                    scalars=False, images=False, logits=True)['loss']

            # clear old gradients
            optimizer.zero_grad()

            # backpropagate the error
            class_loss.backward()

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
