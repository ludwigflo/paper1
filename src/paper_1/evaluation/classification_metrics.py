from snorkel.classification import cross_entropy_with_probs
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from typing import Union
import torch.nn as nn
import seaborn as sn
import pandas as pd
import numpy as np
import inspect
import torch


# noinspection SpellCheckingInspection,PyUnresolvedReferences,PyTypeChecker
class ClassificationMetrics:

    def __init__(self, inv_class_mapping: Union[None, dict] = None, multi_label: bool = False):
        """
        Constructor of the ClassificationMetrics class.

        Parameters
        ----------
        num_classes: number of classes in the classification problem.
        multi_label: Whether to perform multi label classification or not.
        inverse_class_mapping: A mapping from class indices to class names.
        """

        # store properties of the classification problem
        self.inverse_class_mapping = inv_class_mapping
        self.multi_label = multi_label
        self.bce_logits_loss = None
        self.nll_loss = None
        self.bce_loss = None
        self.ce_loss = None

    def _binary_cross_entropy(self, pred: torch.Tensor, tar: torch.Tensor, logits: bool = True) -> torch.Tensor:
        """
        Computes the binary cross entropy between a batch of predictions and a batch of targets.

        Parameters
        ----------
        pred: Computed predictions.
        tar: Groundtrouth labels.
        weight: (Optional) A Tensor, which weights the samples in a batch.
        logits: Whether the outputs are logits or probabilities.

        Returns
        -------
        ce: Binary Crossentropy loss.
        """

        # reshape the predictions and the targets
        pred, tar = pred.view(-1), tar.view(-1)

        # check if the targets are in the correct range
        assert (torch.max(tar) <= 1) and (torch.min(tar) >= 0), "Labels need to be in a range between 0 and 1!"

        # check if the dimensionality is correct
        assert pred.size() == tar.size(), "The size of the predictions needs to be equal to the size of the targets!"

        # compute the binary cross entropy loss
        if logits:

            # initialize the loss function, if it isn't already
            if self.bce_logits_loss is None:
                self.bce_logits_loss = nn.BCEWithLogitsLoss()

            # compute the loss
            ce = self.bce_logits_loss(pred.float(), tar.float())
        else:
            # initialize the loss function, if it isn't already
            if self.bce_loss is None:
                self.bce_loss = nn.BCELoss()

            # compute the loss
            ce = self.bce_loss(pred, tar)
        return ce

    def _multi_class_cross_entropy(self, pred: torch.Tensor, tar: torch.Tensor, logits: bool = True) -> torch.Tensor:
        """
        Computes the cross entropy between a batch of predictions and a batch of targets for a multiclass classification problem.

        Parameters
        ----------
        pred: Computed predictions.
        tar: Groundtrouth labels.
        logits: Whether the predictions are logits or probabilities.

        Returns
        -------
        ce: Crossentropy loss.
        """

        # check if the dimensionality is correct
        assert pred.size()[0] == tar.size()[0], "The number of predictions needs to be equal to the number of targets!"

        # compute the binary cross entropy loss
        if logits:

            # initialize the loss function, if it isn't already
            if self.ce_loss is None:
                self.ce_loss = nn.CrossEntropyLoss()

            # compute the loss
            ce = self.ce_loss(pred, tar)
        else:

            # initialize the loss function, if it isn't already
            if self.nll_loss is None:
                self.nll_loss = nn.NLLLoss()

            # compute the loss
            ce = self.nll_loss(pred, tar)
        return ce

    def _multi_label_cross_entropy(self, pred: torch.Tensor, tar: torch.Tensor, logits: bool = True) -> torch.Tensor:
        """
        Computes the cross entropy between a batch of predictions and a batch of targets for a multilabel classification problem.

        Parameters
        ----------
        pred: Computed predictions.
        tar: Groundtrouth labels.
        logits: Whether the predictions are logits or probabilities.

        Returns
        -------
        ce: Crossentropy loss.
        """

        # check if the dimensionality is correct
        assert pred.size() == tar.size(), "The number of predictions needs to be equal to the number of targets!"

        # compute the loss by considering the predictions for each sample as a set of C independent binary predictions
        ce = self._binary_cross_entropy(pred, tar, logits)
        return ce

    def cross_entropy(self, pred: torch.Tensor, tar: torch.Tensor, logits: bool = True) -> torch.Tensor:
        """
        Computes the cross entropy between a batch of predictions and a batch of targets for classification problem.

        Parameters
        ----------
        pred: Computed predictions.
        tar: Groundtrouth labels.
        logits: Whether the predictions are logits or probabilities.

        Returns
        -------
        ce: Crossentropy loss.
        """

        if self.multi_label:
            ce = self._multi_label_cross_entropy(pred, tar, logits)
        elif pred.size()[1] == 1:
            ce = self._binary_cross_entropy(pred, tar, logits)
        else:
            ce = self._multi_class_cross_entropy(pred, tar, logits)
        return ce

    def l1_loss(self, predictions: torch.Tensor, targets: torch.Tensor, logits: bool = True) -> torch.Tensor:
        """

        Parameters
        ----------
        predictions
        targets
        logits

        Returns
        -------

        """

        if logits:
            predictions = nn.functional.softmax(predictions, dim=1)
        loss = nn.functional.l1_loss(predictions, targets, reduction='mean')
        return loss

    def soft_cross_entropy_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        """

        Parameters
        ----------
        predictions
        targets

        Returns
        -------

        """

        loss = cross_entropy_with_probs(predictions, targets)
        return loss

    def prepare_report_data(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
        """

        Parameters
        ----------
        predictions
        targets

        Returns
        -------

        """

        if self.multi_label:
            predictions = predictions.view(-1)
            predictions = torch.LongTensor(predictions.view(-1) > 0).detach().cpu().numpy()
            targets = targets.view(-1)
        else:
            if predictions.size()[1] > 1:
                predictions = torch.argmax(predictions, dim=1).detach().cpu().numpy()
                if len(targets.size()) > 1:
                    targets = torch.argmax(targets, dim=1)
                targets = targets.detach().cpu().numpy()
            else:
                predictions = torch.LongTensor(predictions.view(-1) > 0).detach().cpu().numpy()
                targets = targets.view(-1).detach().cpu().numpy()

        return predictions, targets

    def compute_classification_report(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """

        Parameters
        ----------
        predictions
        targets

        Returns
        -------

        """

        # convert predictions and targets in a form, such that a report can be computed
        predictions, targets = self.prepare_report_data(predictions, targets)

        # get the names and indices of the labels
        target_names = list(self.inverse_class_mapping.values())
        target_indices = list(self.inverse_class_mapping.keys())
        report_dict = classification_report(targets, predictions, labels=target_indices, target_names=target_names, output_dict=True)

        out_dict = {}
        for label, item in report_dict.items():
            if type(item) == dict:
                for metric, value in item.items():
                    if metric not in out_dict:
                        out_dict[metric] = {}
                    out_dict[metric][label] = value
            else:
                out_dict[label] = item
        return out_dict

    def auc(self, predictions: torch.Tensor, targets: torch.Tensor, logits: bool = True):
        """

        Parameters
        ----------
        predictions
        targets
        logits
        """

        _, targets = self.prepare_report_data(predictions, targets)
        auc_dict = dict()
        if logits:
            predictions = nn.functional.softmax(predictions, dim=1)
        predictions = predictions.detach().cpu().numpy()
        auc_dict['macro avg'] = roc_auc_score(targets, predictions, average='macro', multi_class='ovo')
        auc_dict['weighted avg'] = roc_auc_score(targets, predictions, average='weighted', multi_class='ovo')
        return auc_dict

    def compute_confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """
        Plots a confsion matrix given a batch of predictions and corresponding labels.

        Parameters
        ----------
        predictions: Predictions from a classifier (NxC).
        targets: Groundtrouth labels (N,).

        Returns
        -------
        matrix: Confusion matrix of predictions given corresponding groundtruth values.
        """

        predictions, targets = self.prepare_report_data(predictions, targets)
        matrix = confusion_matrix(targets, predictions, normalize=None)
        return matrix

    def plot_confusion_matrix(self, matrix: np.ndarray, figsize=(15, 12)) -> object:
        """

        Parameters
        ----------
        matrix: Confusion matrix of predictions given corresponding groundtruth values.
        figsize: Size of the plot.
        """

        # sort the class names according to their indices
        name_list = [(class_index, class_name) for class_index, class_name in self.inverse_class_mapping.items()]
        name_list = sorted(name_list, key=lambda x: x[0])
        name_list = [x[1] for x in name_list]

        # get a list of class names in the correct order
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('white')
        df_cm = pd.DataFrame(matrix, name_list, name_list)
        cmap = sn.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
        ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=cmap)
        ax.set(xlabel='Predictions', ylabel='Targets')
        return plt

    def _get_unprotected_class_methods(self) -> list:
        """
        Gets all methods of the class ClassificationMetrics, which are not protected (which do not start with an underscore).

        Returns
        -------
        method_list: List of names of unprotected methods.
        """

        exclude_list = ['plot_confusion_matrix']
        method_list = inspect.getmembers(self, predicate=inspect.ismethod)
        method_list = [x[0] for x in method_list if x[0][0] != '_' and x[0] not in exclude_list]
        return method_list

    def compute_metrics(self, preds: torch.Tensor, tars: torch.Tensor, fig_size: tuple = None, loss: Union[None, str] = True,
                        scalars: bool = True, images: bool = True, logits: bool = True) -> dict:
        """
        """

        output_dict = dict()

        # if scalars should be reported
        if scalars:
            # compute a classification report
            output_dict['scalars'] = self.compute_classification_report(preds, tars)

            # compute the area under the curve
            output_dict['scalars']['auc'] = self.auc(preds, tars, logits)

        # if images should be reported
        if images:
            # commpute a confusion matrix, if we do not perform multilabel classification
            if not self.multi_label:
                m = self.compute_confusion_matrix(preds, tars)
                if fig_size is None:
                    f = self.plot_confusion_matrix(m)
                else:
                    f = self.plot_confusion_matrix(fig_size, fig_size)
                output_dict['images'] = dict()
                output_dict['images']['Confusion Matrix'] = f
                plt.close()

        # if the loss should be reported
        if loss is not None:
            if loss == 'cross_entropy':
                loss = self.cross_entropy(preds, tars, logits)
            elif loss == 'soft_cross_entropy':
                loss = self.soft_cross_entropy_loss(preds, tars)
            elif loss == 'l1':
                loss = self.l1_loss(preds, tars, logits)
            else:
                loss = None
            output_dict['loss'] = loss
        return output_dict
