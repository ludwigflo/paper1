from paper_1.data.data_loader import load_val_data, load_train_data, sequential_data_loader, random_data_loader
from paper_1.evaluation.classification_metrics import ClassificationMetrics
from paper_1.evaluation.eval_utils import init_metrics_object
from paper_1.model.model_utils import initialize_model
from paper_1.utils import create_experiment_directory
from paper_1.model.embeddings import EmbeddingModule
from torch.utils.tensorboard import SummaryWriter
from .train import train_epoch, validation_epoch
from paper_1.utils import read_parameter_file
import paper_1.model.tokenizer as tokenizer
from os.path import dirname, abspath
from torch.optim import Adam
from typing import Generator
import torch.nn as nn
import numpy as np
import random
import torch
import math
import os


def train(model: nn.Module, optimizer: Adam, metric_object: ClassificationMetrics, num_train_iter: int,
          metric_params: dict, train_loader: Generator, val_loader_list: list, src_domain: str, writer: SummaryWriter,
          num_val_iter_list: list, val_domain_list, num_epochs: int, exp_dir: str):
    """"""

    # initialize the f1 score for the currently best model
    f1_best = -math.inf

    for epoch in range(num_epochs):
        print()
        print('Epoch: {:3d}'.format(epoch))

        train_metrics = train_epoch(model, optimizer, metric_object, num_train_iter, metric_params, train_loader, src_domain)
        val_metrics = validation_epoch(model, metric_object, num_val_iter_list, metric_params, val_loader_list, val_domain_list)

        # store the model, if its performance is better than the previously best model
        if val_metrics['scalars']['f1-score/' + src_domain + '/validation']['macro avg'] > f1_best:
            f1_best = val_metrics['scalars']['f1-score/' + src_domain + '/validation']['macro avg']
            torch.save(model, exp_dir + 'f1_best.pt')

        # log the metrics to tensorboard
        for key, train_value in train_metrics['scalars'].items():
            if type(train_value) == dict:
                for sub_key, sub_value in train_value.items():
                    key_final = key + '/' + sub_key
                    writer.add_scalar(key_final, sub_value, epoch)
            else:
                writer.add_scalar(key, train_value, epoch)
        writer.flush()

        for key, val_value in val_metrics['scalars'].items():
            if type(val_value) == dict:
                for sub_key, sub_value in val_value.items():
                    key_final = key + '/' + sub_key
                    writer.add_scalar(key_final, sub_value, epoch)
            else:
                writer.add_scalar(key, val_value, epoch)
        writer.flush()


# noinspection PyTupleAssignmentBalance
def main(main_params: dict, data_params: dict, metric_params: dict, model_params: dict, parent_dir, source_domain: str):

    # clear the cuda memory
    torch.cuda.empty_cache()

    # read the train params
    num_train_iter = main_params['num_train_iter']
    experiment_id = main_params['experiment_id']
    num_epochs = main_params['num_epochs']
    base_dir = main_params['base_dir']

    # load the train and validation data
    data_train = load_train_data(data_params, source_domain, 'None')
    data_list_val = load_val_data(data_params)
    num_val_iter_list = [df.shape[0] for df in data_list_val]

    # extract the remaining data loader params
    batch_size = data_params['data_loader']['batch_size']
    balance_keys = data_params['data_loader']['balance_keys']
    balance_keys = None if balance_keys == 'None' else balance_keys
    train_loader = random_data_loader(data_train, balance_keys, batch_size)
    validation_domains = data_params['data']['validation']['validation_domains']
    val_loader_list = [sequential_data_loader(data_frame) for data_frame in data_list_val]

    # initialize the metrics object
    metric_object = init_metrics_object(metric_params)

    # create a model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = initialize_model(model_params, parent_dir, device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # create an optimizer for the model
    optimizer = Adam(model.parameters(), lr=4e-5, betas=(0.9, 0.999))

    # create a directory for the current experiments
    file_names_params = os.listdir(parent_dir + '/parameters/')
    file_names_params = [parent_dir + '/parameters/' + x for x in file_names_params]
    file_names_baseline = os.listdir(parent_dir + '/baseline/')
    file_names_baseline = [parent_dir + '/baseline/' + x for x in file_names_baseline]
    file_names = []
    file_names.extend(file_names_params)
    file_names.extend(file_names_baseline)
    file_names = [x for x in file_names if not os.path.isdir(x)]
    print(file_names)

    val_fold = data_params['data']['val_fold']
    exp_dir = create_experiment_directory(base_dir, source_domain, 'None', val_fold, file_names, experiment_id)

    # create a tensorboard writer
    writer = SummaryWriter(exp_dir)

    train(model, optimizer, metric_object, num_train_iter, metric_params, train_loader, val_loader_list,
          source_domain, writer, num_val_iter_list, validation_domains, num_epochs, exp_dir)
    del model
    del optimizer
    del discriminator


if __name__ == '__main__':

    # set the seed for reproducability
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # get the current and parent directory
    current_file = abspath(__file__)
    current_dir = dirname(current_file)
    parent_dir = dirname(current_dir)
    metric_param_file = parent_dir + '/parameters/metric_params.yaml'
    model_param_file = parent_dir + '/parameters/model_params.yaml'
    data_param_file = parent_dir + '/parameters/data_params.yaml'
    main_param_file = current_dir + '/main_params.yaml'

    # load the parameters
    metric_params = read_parameter_file(metric_param_file)
    model_params = read_parameter_file(model_param_file)
    main_params = read_parameter_file(main_param_file)
    data_params = read_parameter_file(data_param_file)

    # define the domains, on which the models should be trained
    source_domains = ['Religion']

    for source_domain in source_domains:
        main(main_params, data_params, metric_params, model_params, parent_dir, source_domain)
