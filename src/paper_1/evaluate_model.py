from paper_1.evaluation.classification_metrics import ClassificationMetrics
from tensorflow.python.summary.summary_iterator import summary_iterator
from paper_1.data.data_loader import read_tsv_file
import matplotlib.pyplot as plt
from typing import List, Union
import seaborn as sb
import pandas as pd
import numpy as np
import torch
import csv
import os


def create_tag_dict(iterator: summary_iterator) -> dict:
    """
    Creates a dictionary, which maps tags to a tuple consisting of time steps and values.

    Parameters
    ----------
    iterator: Tensorboard iterator, which extracts scalar values for tensorboard logfiles.

    Returns
    -------
    tag_dict: Dictionary, which maps tags to value.
    """

    tag_dict = {}

    for e in iterator:
        step = e.step
        summary = e.summary
        tag = None
        value = None
        for val in summary.value:
            value = val.simple_value
            tag = val.tag

        if tag not in tag_dict:
            tag_dict[tag] = [(step, value)]
        else:
            tag_dict[tag].append((step, value))
    return tag_dict


def create_csv(csv_path: str, tag_dict: dict) -> None:
    """
    Create a csv file, based on the extracted tag dict from tensorboard.

    Parameters
    ----------
    csv_path: Path to the csv file.
    tag_dict: Dictionary, which maps tensorboard tags to their values (time steps and metrics)
    """

    # create the header of the csv file
    header = ['step']
    header.extend(list(tag_dict.keys()))
    header = [x for x in header if x is not None]
    header = [x for x in header if x.split(' ')[0] != 'pseudo_label']

    # convert the data, such that they can be written into a csv file
    time_steps = None
    csv_list = []
    for tag, values in tag_dict.items():
        if tag is not None:
            t = tag.split(' ')[0]
            if t != 'pseudo_label':
                if values is not None:
                    if time_steps is None:
                        time_steps = [x[0] for x in values]
                        csv_list.append(time_steps)
                    values = [x[1] for x in values]
                    csv_list.append(values)

    # write the csv file
    with open(csv_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

        for line in zip(*csv_list):
            csv_writer.writerow(line)


def get_metric_columns(data_frame: pd.DataFrame, metric: str, scheme: Union[None, str], mode: str = 'validation') -> tuple:
    """
    Returns a list of columns, in which the training results of a desired metric and a desired mode are stored.

    Parameters
    ----------
    data_frame: Header of a csv file, in which all results are stored.
    metric: Metric, which should be extracted.
    scheme: Current scheme of the metric, e.g. macro_avg, which should be extracted.
    mode: Mode, which should be extracted (either 'validation' or 'train').

    Returns
    -------
    column_idx_list: Indices of the columns, in which the desired properties are stored.
    target_list: list of target groups corresponding to the column indices.
    """

    header = data_frame.columns

    column_idx_list = []
    target_list = []
    for i, tag in enumerate(header):

        if i > 0:
            # split the current tag in the header
            tag = tag.split('/')

            if len(tag) == 4:
                # get the metric and its corresponding scheme
                metric_act = tag[0]
                target_act = tag[1]
                scheme_act = tag[3]
                mode_act = tag[2]

                if metric_act == metric and scheme_act == scheme and mode_act == mode:
                    column_idx_list.append(i)
                    target_list.append(target_act)
            else:
                # get the metric and its corresponding scheme
                metric_act = tag[0]
                target_act = tag[1]
                mode_act = tag[2]

                if metric_act == metric and mode_act == mode:
                    column_idx_list.append(i)
                    target_list.append(target_act)
    return column_idx_list, target_list


def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Reads the results from a csv file.

    Parameters
    ----------
    file_path: Path to the file, in which the results are stored.

    Returns
    -------
    header: Header of the csv file containing the tags of the columns.
    columns: Results, which have been stored in the csv file.
    """

    df = read_tsv_file(file_path, delimiter=',')
    return df


def get_max_val(column_index: int, data_frame: pd.DataFrame) -> tuple:
    """
    Returns the largest value in a given csv column as well as its corresponding time step.

    Parameters
    ----------
    column_index: Index of the column, in which the max val should be determined.
    data_frame: Columns, from which the data should be retrieved.

    Returns
    -------
    max_val: Largest value in the provided column.
    time_step: Time step, in which the largest value occurred.
    """

    # get the current column
    column = data_frame.iloc[:, column_index].tolist()

    # get the max val in the selected column
    max_val = max(column)

    # get the time step of the max val
    time_step = np.argmax(column)
    return max_val, time_step


def select_results_by_time_step(column_indices: list, data_frame: pd.DataFrame, time_step: int) -> list:
    """
    Selects the results from desired columns in a csv file at a given time step.

    Parameters
    ----------
    column_indices: Indices of the columns, which should be evaluated.
    data_frame: Columns, containing the results.
    time_step: Time step, from which the results should be evaluated.

    Returns
    -------
    result_list: List of results from the columns in the csv file at a certain time step.
    """

    result_list = []
    for column_index in column_indices:

        column = data_frame.iloc[:, column_index].tolist()
        result_list.append(column[time_step])
    return result_list


def compute_single_results(base_path: str, file_name: str, selection_metric: str, selection_scheme: Union[None, str],
                           selection_mode: str, selection_domain: str, result_scheme: str, result_mode: str, result_metric: str):
    """

    Parameters
    ----------
    base_path
    file_name
    selection_metric
    selection_mode
    selection_scheme
    selection_domain
    result_scheme
    result_mode
    result_metric
    """

    path = base_path + file_name
    csv_path = base_path + 'results.csv'

    # read the data from the tensorboard summary writer file
    iterator = summary_iterator(path)
    tag_dict = create_tag_dict(iterator)

    # create a csv file for storing the results
    create_csv(csv_path, tag_dict)

    # read the results
    data_frame = read_csv_file(csv_path)

    # get the desired results in columns
    column_indices, target_groups = get_metric_columns(data_frame, selection_metric, selection_scheme, mode=selection_mode)

    # determine the time step of the best results of the desired result
    selection_col_index = target_groups.index(selection_domain)

    _, time_step = get_max_val(column_indices[selection_col_index], data_frame)

    # get the targets and columns of the metrics, which should be reported
    column_indices, target_groups = get_metric_columns(data_frame, result_metric, result_scheme, mode=result_mode)
    results = select_results_by_time_step(column_indices, data_frame, time_step)

    result_dict = {}
    for key, value in zip(target_groups, results):
        result_dict[key] = value
    return result_dict


def print_extended_results(exp_path: str, validation_folds: list, selection_metric: str, selection_scheme: Union[None, str],
                           selection_mode: str, result_metric: str, result_scheme: str, result_mode: str, train_domains: list,
                           target_domains: list, cur_num=None):
    if cur_num is not None:
        exp_path += str(cur_num) + '/'
    avg_results = {}
    for fold in validation_folds:
        for source_domain in train_domains:

            if source_domain not in avg_results:
                avg_results[source_domain] = {}

            target_domain_list = os.listdir(exp_path + source_domain)
            target_domain_list = [x for x in target_domain_list if x in target_domains]
            for i, target_domain in enumerate(target_domain_list):
                if target_domain not in avg_results[source_domain]:
                    avg_results[source_domain][target_domain] = {}

                path = exp_path + source_domain + '/' + target_domain + '/' + str(fold) + '/'

                files = os.listdir(path)

                file = [x for x in files if x.startswith('events.out')][0]
                results = compute_single_results(path, file, selection_metric, selection_scheme, selection_mode,
                                                 source_domain, result_scheme, result_mode, result_metric)

                for validation_domain, result in results.items():
                    if validation_domain in avg_results[source_domain][target_domain]:
                        avg_results[source_domain][target_domain][validation_domain] += result / len(validation_folds)
                    else:
                        avg_results[source_domain][target_domain][validation_domain] = result / len(validation_folds)

    pd_list = []
    for source_domain in avg_results:
        for target_domain in avg_results[source_domain]:
            for validation_domain in avg_results[source_domain][target_domain]:
                result = avg_results[source_domain][target_domain][validation_domain]
                pd_dict = {'Source': source_domain, 'Target': target_domain,
                           'Validation': validation_domain, 'Result': result}
                pd_list.append(pd_dict)

    df = pd.DataFrame(pd_list)
    print(df)
    plt.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.savefig(exp_path + result_metric + '_' + result_scheme + '.png')


if __name__ == '__main__':

    exp_path = 'experiments/mixup/3/'
    train_domains = ['Race', 'Religion', 'Sexual Orientation']
    val_domains = ['Race', 'Religion', 'Sexual Orientation']
    target_domains = ['Race', 'Religion', 'Sexual Orientation']
    validation_folds = [0]
    selection_metric = 'f1-score'
    selection_scheme = 'macro avg'
    selection_mode = 'validation'
    result_metric = 'f1-score'
    result_scheme = 'macro avg'
    result_mode = 'validation'

    print_extended_results(exp_path, validation_folds, selection_metric, selection_scheme, selection_mode,
                           result_metric, result_scheme, result_mode, train_domains, target_domains, None)
