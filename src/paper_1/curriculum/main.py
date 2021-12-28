from paper_1.data.data_loader import load_val_data, load_train_data, sequential_data_loader, random_data_loader
from paper_1.utils import read_parameter_file, create_experiment_directory
from paper_1.evaluation.eval_utils import init_metrics_object
from paper_1.baseline.main import train as baseline_train
from paper_1.model.model_utils import initialize_model
from torch.utils.tensorboard import SummaryWriter
from train import select_splitted_pseudo_labels
from os.path import dirname, abspath
from torch.optim import Adam
import pandas as pd
import numpy as np
import random
import torch
import os


def main(main_params: dict, data_params: dict, metric_params: dict, model_params: dict,
         parent_dir, source_domain: str, target_domain: str):

    # clear the cuda memory
    torch.cuda.empty_cache()

    # get the current validation fold
    val_fold = data_params['data']['val_fold']

    # read the train params
    num_train_iter = main_params['num_train_iter']
    experiment_id = main_params['experiment_id']
    num_epochs = main_params['num_epochs']
    quantiles = main_params['quantiles']
    model_dir = main_params['model_dir']
    base_dir = main_params['base_dir']

    # get the data loader parameters
    balance_keys = data_params['data_loader']['balance_keys']
    batch_size = data_params['data_loader']['batch_size']

    # load the data
    data_train_src, data_train_tar = load_train_data(data_params, source_domain, target_domain)
    data_list_val = load_val_data(data_params)
    num_val_iter_list = [df.shape[0] for df in data_list_val]
    validation_domains = data_params['data']['validation']['validation_domains']
    val_loader_list = [sequential_data_loader(data_frame) for data_frame in data_list_val]

    # load a pre trained model
    model_path = model_dir + source_domain + '/' + 'None' + '/' + str(val_fold) + '/f1_best.pt'

    # load a previously stored model, which is the init point for curriculum labeling
    pretrained_model = torch.load(model_path)
    mapping = metric_params['inverse_class_mapping']

    # initialize the metrics object
    metric_object = init_metrics_object(metric_params)

    # create a directory for the current experiments
    file_names_params = os.listdir(parent_dir + '/parameters/')
    file_names_params = [parent_dir + '/parameters/' + x for x in file_names_params]
    file_names_baseline = os.listdir(parent_dir + '/baseline/')
    file_names_baseline = [parent_dir + '/baseline/' + x for x in file_names_baseline]

    file_names = []
    file_names.extend(file_names_params)
    file_names.extend(file_names_baseline)
    file_names = [x for x in file_names if not os.path.isdir(x)]

    val_fold = data_params['data']['val_fold']
    exp_base_dir = create_experiment_directory(base_dir, source_domain, target_domain, val_fold, file_names, experiment_id)

    for quantile in quantiles:

        exp_dir = exp_base_dir + str(quantile) + '/'
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # create a tensorboard writer
        writer = SummaryWriter(exp_dir)

        # create data loader with current pseudo labels
        data_frame_pseudo = select_splitted_pseudo_labels(pretrained_model, data_train_tar, quantile, mapping)

        # delete the previously trained model, as it is no longer in use
        del pretrained_model

        # create the train data loader
        data_train = pd.concat([data_train_src, data_frame_pseudo])
        train_loader = random_data_loader(data_train, balance_keys, batch_size)

        # initialize a new model to train it from scratch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = initialize_model(model_params, parent_dir, device)
        model.cuda()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # create an optimizer for the model
        optimizer = Adam(model.parameters(), lr=4e-5, betas=(0.9, 0.999))

        # train the newly created model from scratch
        baseline_train(model, optimizer, metric_object, num_train_iter, metric_params, train_loader, val_loader_list,
                       source_domain, writer, num_val_iter_list, validation_domains, num_epochs, exp_dir)

        # update the pretrained model
        pretrained_model = model

    del model
    del optimizer


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
    source_domains = ['Race', 'Religion', 'Sexual Orientation']
    target_domains = ['Race', 'Religion', 'Sexual Orientation']

    for source_domain in source_domains:
        for target_domain in target_domains:
            if source_domain != target_domain:
                main(main_params, data_params, metric_params, model_params, parent_dir, source_domain, target_domain)
