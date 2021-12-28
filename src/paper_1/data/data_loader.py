from typing import List, Generator, Tuple, Union, Dict
from paper_1.utils import read_parameter_file
from ast import literal_eval
from copy import deepcopy
import pandas as pd
import numpy as np


def literal_return(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as _:
        return val


def select_row(row_params: tuple):
    labels = row_params[0]
    selection_labels = row_params[1]

    if type(labels) == tuple:
        for label in labels:
            if label in selection_labels:
                return True
        return False
    else:
        if labels in selection_labels:
            return True
        else:
            return False


def read_tsv_file(file_path: str, delimiter: str = '\t', columns: Union[tuple, None] = None, literal_columns: Union[list, None] = None,
                  excluding_dict: Union[Dict, None] = None, including_dict: Union[Dict, None] = None) -> pd.DataFrame:
    """
    Reads a tsv file and returns its content as list.

    Parameters
    ----------
    file_path: File path, in which the tsv file is stored.
    delimiter: Delimiter of the tsv file.
    columns: Columns, which should be loaded.
    literal_columns: Columns, which need to be loaded with literal eval.
    excluding_dict: Dictionary, which maps columns to lists of labels, which should excluded from the dataframe.
    including_dict: Dictionary, which maps columns to lists of labels, which should not be excluded from the dataframe.

    Returns
    -------
    data_list: List of data samples.
    """

    # read the tsv dataset file and select the provided columns
    df = pd.read_csv(file_path, delimiter=delimiter)

    # apply literal eval function to the data frame for specified columns
    if literal_columns is not None:
        for literal_column in literal_columns:
            df[literal_column] = df[literal_column].map(lambda val: literal_return(val))

    # exclude samples, which contain labels in certain columns, which shuld be excluded from the data frame
    if excluding_dict is not None:
        for column_name, labels in excluding_dict.items():
            df = df[~df[column_name].map(lambda x: select_row((x, labels)))]

    # store only those samples, which match the conditions in the include_dict
    if including_dict is not None:
        for column_name, labels in including_dict.items():
            df = df[df[column_name].map(lambda x: select_row((x, labels)))]

    if columns is not None:
        df = df[columns]
    return df


def select_label_combination(row_params: tuple):

    row_value = tuple(row_params[0])
    record = row_params[1]
    return row_value == record


def compute_sampling_dict(data_frame: pd.DataFrame, balance_keys: Union[None, list] = None) -> dict:
    """
    Dictionary, which is the basis for sampling data points. It maps specified label combinations to data IDs.
    The specified labels are sampled with equal probabilities.

    Parameters
    ----------
    data_frame: Dataset, from which data points should be sampled.
    balance_keys: The keys (columns in the csv file), for which its values should be sampled with equal probability.

    Returns
    -------
    sampling_dict: Dictionary, which maps keys to data points belonging to these keys.
    """

    # initialize the sampling dictionary
    sampling_dict = dict()

    # if we do not want to sample keys in a balanced manner
    if balance_keys is None:
        sampling_dict['all'] = data_frame

    # if we want to sample some provided keys in a balanced manner
    else:
        sub_df = data_frame[balance_keys]
        records = [tuple(x) for x in sub_df.to_records(index=False)]
        unique_records = []
        for record in records:
            if record not in unique_records:
                unique_records.append(record)
        for record in unique_records:
            df_rec = data_frame[sub_df.apply(lambda x: select_label_combination((x, record)), axis=1)]
            sampling_dict[record] = df_rec
    return sampling_dict


def sample_data(sampling_dict: dict, batch_size: int) -> tuple:

    # get the keys from the sampling dict
    dict_keys = list(sampling_dict.keys())
    key_indices = list(range(len(dict_keys)))

    # sample each sample in the batch
    batch_list = []
    for _ in range(batch_size):

        # sample a dictionary key
        key_index = np.random.choice(key_indices, size=1, replace=True)[0]
        key = dict_keys[key_index]

        # get the corresponding sub dataframe
        sampled_df = sampling_dict[key]

        # sample a data sample from the sub dataframe
        sample = sampled_df.sample(n=1).to_records(index=False).tolist()[0]
        batch_list.append(sample)
    batch_list = list(zip(*batch_list))
    return batch_list


def random_data_loader(data_frame: pd.DataFrame, balance_keys: Union[list, None] = None, batch_size: int = 1):

    # get the number of data points
    num_data_points = len(data_frame.index)

    # copmpute a dictionary, which is used for sampling the data samples
    sampling_dict = compute_sampling_dict(data_frame, balance_keys=balance_keys)

    sample_count = 0
    while True:

        # sample a data batch
        data_batch = sample_data(sampling_dict, batch_size)

        # increase the counter of the samples
        sample_count += batch_size

        # check if the epoch is done
        done = sample_count >= num_data_points

        if done:
            sample_count = 0

        # return the samples and the epoch state
        yield data_batch, done


def sequential_data_loader(data_frame: pd.DataFrame):

    # get the number of data points
    num_data_points = len(data_frame.index)
    data_frame = data_frame.to_records(index=False)

    while True:
        sample_count = 0

        for sample in data_frame:
            sample_count += 1
            done = sample_count == num_data_points
            if done:
                sample_count = 0

            sample = tuple([[x] for x in sample])
            yield sample, done


def load_val_data(data_params: dict) -> List[pd.DataFrame]:

    # extract the relevant information
    validation_domains = data_params['data']['validation']['validation_domains']
    literal_columns = data_params['data']['validation']['literal_columns']
    data_path = data_params['data']['validation']['data_path']
    columns = data_params['data']['validation']['columns']
    domain_list = data_params['data']['domain_list']
    val_fold = data_params['data']['val_fold']

    data_list = []
    for validation_domain in validation_domains:
        exclude_domains = deepcopy(domain_list)
        exclude_domains.remove(validation_domain)
        include_dict = {
            'targets': [validation_domain],
            'fold': [val_fold],
        }
        exclude_dict = {
            'targets': exclude_domains
        }
        # load the source domain data
        df = read_tsv_file(data_path, literal_columns=literal_columns, excluding_dict=exclude_dict,
                           including_dict=include_dict, columns=columns)
        data_list.append(df)
    return data_list


def load_train_data(data_params: dict, src_domain: str, target_domain: str) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:

    # extract the relevant information
    literal_columns = data_params['data']['train']['literal_columns']
    data_path = data_params['data']['train']['data_path']
    columns = data_params['data']['train']['columns']
    domain_list = data_params['data']['domain_list']
    val_fold = data_params['data']['val_fold']

    # convert data types
    if target_domain == 'None':
        target_domain = None

    # compute the dictionaries, which determine which data points include and exclude
    exclude_domains = deepcopy(domain_list)
    exclude_domains.remove(src_domain)
    include_dict = {
        'targets': [src_domain]
    }
    exclude_dict = {
        'fold': [val_fold],
        'targets': exclude_domains
    }

    # load the source domain data
    df_source = read_tsv_file(data_path, literal_columns=literal_columns, excluding_dict=exclude_dict,
                              including_dict=include_dict, columns=columns)

    # if a target domain is specified, also load the data points from the target domain
    if target_domain is not None:

        # compute the dictionaries, which determine which data points include and exclude
        exclude_domains = deepcopy(domain_list)
        exclude_domains.remove(target_domain)
        include_dict = {
            'targets': [target_domain]
        }
        exclude_dict = {
            'fold': [val_fold],
            'targets': exclude_domains
        }

        # load the target domain data
        df_target = read_tsv_file(data_path, literal_columns=literal_columns, excluding_dict=exclude_dict,
                                  including_dict=include_dict, columns=columns)
        return df_source, df_target
    else:
        return df_source


if __name__ == '__main__':

    parameter_file = '/paper_1/parameters/data_params.yaml'
    data_params = read_parameter_file(parameter_file)
    source_domain = 'Race'
    target_domain = 'None'
    df_src = load_train_data(data_params, source_domain, target_domain)
    print(df_src)

    df_val = load_val_data(data_params)
    for df in df_val:
        print()
        print(df)
