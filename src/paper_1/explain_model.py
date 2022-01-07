from nlp_models.classification.lstm_attn_classifier import LSTMAttnClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from lime.lime_text import LimeTextExplainer
from data_loader import read_tsv_file
from html2image import Html2Image
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil
import torch
import umap
import sys
import os


class LimeWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, data: list):
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(data, tokenize=True, logits=False)
            predictions = predictions.detach().cpu().numpy()
        return predictions


class LimeTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data: list):
        data_tokenized = self.tokenizer.tokenize([data], token_ids=False)
        token_string = ' '.join(data_tokenized[0])
        return token_string, data_tokenized


def load_data(tsv_file_path: str, targets: str, fold: list, label: list, rationals: bool = False):

    # read the provided tsv file
    header, data_list = read_tsv_file(tsv_file_path)

    # get the index position of the required properties
    data_idx = header.index('text_data')
    target_idx = header.index('targets')
    label_idx = header.index('labels')
    fold_idx = header.index('fold')
    id_idx = header.index('ID')

    data_list_selected, label_list_selected, id_list, rational_list = [], [], [], []
    for sample in data_list:
        if targets in sample[target_idx] and len(sample[target_idx]) == 1 and sample[label_idx] in label and sample[fold_idx] in fold:
            data_list_selected.append(sample[data_idx])
            label_list_selected.append(sample[label_idx])
            id_list.append(sample[id_idx])
            if rationals:
                rationals_idx = header.index('rationals')
                rational_list.append(sample[rationals_idx])
    return data_list_selected, label_list_selected, id_list, rational_list


def predict_data(data_list: list, id_list: list, model: LSTMAttnClassifier, inv_class_mapping: dict, class_label: str) -> list:

    with torch.no_grad():

        model.eval()

        # compute a list of predictions
        prediction_list = [model([sample], logits=False).detach().cpu() for sample in data_list]

        # get the index of the correct class
        true_idx = inv_class_mapping[class_label]

        # divide the predictions in true positives and false negatives
        true_positives, false_negatives, true_ids, false_ids = [], [], [], []
        for i, prediction in enumerate(prediction_list):

            # get the index of the predicted class
            predicted_class_idx = torch.argmax(prediction).item()

            # if the index is correct, then append the value to the true positives, ele to the false negatives
            if predicted_class_idx == true_idx:
                true_positives.append((data_list[i], prediction[0, true_idx].item(), predicted_class_idx, id_list[i], true_idx))
                true_ids.append(id_list[i])

            else:
                false_negatives.append((data_list[i], prediction[0, true_idx].item(), predicted_class_idx, id_list[i], true_idx))
                false_ids.append(id_list[i])
    return true_positives, false_negatives, true_ids, false_ids


def compute_probabilities(model, data_list, label_list):
    probab_list = [torch.softmax(model([data]), dim=1) for data in data_list]
    prediction_list = [torch.argmax(x[0, ...], dim=0) for x in probab_list]

    correct_list, incorrect_list = [], []
    for probab, prediction, label in zip(probab_list, prediction_list, label_list):
        if prediction.item() == label:
            correct_list.append(probab[0, label])
        else:
            incorrect_list.append(probab[0, label])
    return correct_list, incorrect_list


def get_average_thresholds(base_dir: str, tsv_file_path: str, approach: str, inv_class_mapping: dict, exp_id: int):

    exp_dir = base_dir + approach + '/' + str(exp_id) + '/'
    label = ['normal', 'offensive', 'hatespeech']

    # get the source domains
    source_domain_list = os.listdir(exp_dir)
    source_domain_list = [x for x in source_domain_list if os.path.isdir(os.path.join(exp_dir, x))]
    print(source_domain_list)

    # iterate through the source domains
    for source_domain in source_domain_list:
        source_domain_path = exp_dir + source_domain + '/'

        # get the target domains
        target_domain_list = os.listdir(source_domain_path)
        target_domain_list = [x for x in target_domain_list if os.path.isdir(os.path.join(exp_dir, x))]
        print(target_domain_list)

        # iterate through the target domains
        for target_domain in target_domain_list:
            target_domain_path = source_domain_path + target_domain + '/'

            # get the validation folds
            validation_fold_list = os.listdir(target_domain_path)
            validation_fold_list = [x for x in validation_fold_list if os.path.isdir(os.path.join(target_domain_path, x))]

            # iterate through the validation folds
            for fold in validation_fold_list:
                fold_path = target_domain_path + fold + '/'

                # load the model stored in the current fold
                model = torch.load(fold_path + 'f1_best.pt')
                model.cpu()

                # load source domain data
                data_list, label_list, id_list, rational_list = load_data(tsv_file_path, source_domain, [int(fold)],
                                                                          label, rationals=False)
                label_list = [inv_class_mapping[x] for x in label_list]

                correct_list, incorrect_list = compute_probabilities(model, data_list, label_list)


def process_explanations(explainations, predicted_class, tokens: list):
    e = explainations.as_map()[predicted_class]
    e = sorted(e, key=lambda tup: tup[0])
    indices = [x[0] for x in e]
    weights = [x[1] for x in e]
    min_val, max_val = min(weights), max(weights)
    max_val = max_val if max_val > abs(min_val) else abs(min_val)
    weights = [0.85 * x / max_val for x in weights]
    weight_list = []
    for i in range(len(tokens[0])):
        if i in indices:
            idx = indices.index(i)
            weight_list.append(weights[idx])
        else:
            weight_list.append(0.0)
    return weight_list


def compute_explainations(data_list: str, lime_tokenizer, lime_explaner, model_wrapped):

    word_batch_list, weight_batch_list, probabilities, data_ids, exp_list = [], [], [], [], []

    for i, sample in enumerate(data_list):
        print(i)
        text_data = sample[0]
        data_id = sample[3]
        true_label_idx = sample[-1]

        text_data, tokens = lime_tokenizer(text_data)
        explainations = lime_explaner.explain_instance(text_data, model_wrapped, labels=[true_label_idx])
        exp_list.append(explainations)
        data_ids.append(data_id)
    return word_batch_list, weight_batch_list, probabilities, data_ids, exp_list


def visualize_html(file_name: str, words: list, weight_list: list):

    template = '<span class="barcode"; style="font_size:40.Opt; color: black; background-color: rgba{}">{}</span>'
    colored_string = ''
    for word, weight in zip(words, weight_list):
        print(weight)
        color = (255, 0, 0, weight) if weight > 0 else (0, 255, 0, (-1) * weight)
        print(color)
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')

    # or simply save in an html file and open in browser
    with open(file_name, 'w') as f:
        f.write(colored_string)


def visualize_validation_samples(experiment_dir: str, target_group: str, class_label: str, positive: bool,
                                 word_batch_list: list, weight_batch_list: list, probabilities: list, data_ids: list):
    end_dir = 'tp' if positive else 'fn'
    total_path = experiment_dir + target_group + '/' + class_label + '/' + end_dir + '/'

    html_path = total_path + 'html/'
    png_path = total_path + 'png/'

    if not os.path.exists(html_path):
        os.makedirs(html_path)
    if not os.path.exists(png_path):
        os.makedirs(png_path)
    for data_id, probability, words, weights in zip(data_ids, probabilities, word_batch_list, weight_batch_list):
        html_file = html_path + str(data_id) + '__' + str(probability) + '.html'
        visualize_html(html_file, words[0], weights)

    html_to_png(html_path, png_path)


def html_to_png(html_path: str, png_path: str):

    if not os.path.exists(png_path):
        os.makedirs(png_path)

    html_files = os.listdir(html_path)
    hti = Html2Image(output_path=png_path)

    for html_file in html_files:
        file_name = html_file.split('.html')[0]
        hti.screenshot(html_file=html_path + html_file, save_as=file_name + '.png')


def load_image(img_path: str):
    img = Image.open(img_path)

    img = np.asarray(img)
    img_rows = []
    for row in range(img.shape[0]):

        zeros = True
        r = img[row, ...].tolist()
        for pixel in r:
            if sum(pixel) != 0:
                zeros = False
                break
        if not zeros:
            img_rows.append(img[row, ...])
    img = np.stack(img_rows)
    return img


def select_images(img_ids: list, base_path: str):

    img_names = os.listdir(base_path)
    img_names = [x for x in img_names if int(x.split('__')[0]) in img_ids]
    return img_names


def create_img_plot(img_path: str, base_path: str, img_ids: list):
    img_names = select_images(img_ids, base_path)

    img_list = []
    for img in [load_image(base_path + img_name) for img_name in img_names]:
        img_list.append(img)

    img = Image.fromarray(np.concatenate(img_list, axis=0))
    plt.figure(figsize=(16, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(img_path)


def find_common_explainations(png_paths: list):

    id_dict = {}
    for png_path in png_paths:
        positives = os.listdir(png_path + 'true_positives/')
        for data in positives:
            data_id = data.split('.')[0]
            if data_id in id_dict:
                id_dict[data_id].append((png_path + 'true_positives/' + data, 'positive'))
            else:
                id_dict[data_id] = [(png_path + 'true_positives/' + data, 'positive')]
        negatives = os.listdir(png_path + 'false_negatives/')
        for data in negatives:
            data_id = data.split('.')[0]
            if data_id in id_dict:
                id_dict[data_id].append((png_path + 'false_negatives/' + data, 'negative'))
            else:
                id_dict[data_id] = [(png_path + 'false_negatives/' + data, 'negative')]
    return id_dict


def get_data_ids(path: str):

    files = os.listdir(path)
    ids = [file.split('__')[0] for file in files]
    return ids, files


def find_common_data_ids(id_list_1, id_list_2):
    common_ids = [data_id for data_id in id_list_1 if data_id in id_list_2]
    return common_ids


def get_data_by_id(id_list, path: str):
    ids, files = get_data_ids(path)
    index_list = [ids.index(data_id) for data_id in id_list]
    file_list = [files[index] for index in index_list]
    return file_list


def store_common_explainations(id_dict, name_list, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for data_id, png_paths in id_dict.items():
        for png_path, out_name in zip(png_paths, name_list):
            path = png_path[0]
            label = png_path[1]

            file_name = out_name + '_' + label

            if not os.path.exists(output_path + str(data_id) + '/'):
                os.makedirs(output_path + str(data_id) + '/')
            shutil.move(path, output_path + str(data_id) + '/' + file_name + '.png')


def compute_lime_explainations(output_path, model_path, class_mapping, inv_class_mapping, label):

    html_path = output_path + 'html/'
    png_path = output_path + 'png/'

    fn = 'false_negatives/'
    tp = 'true_positives/'

    if not os.path.exists(html_path + tp):
        os.makedirs(html_path + tp)
    if not os.path.exists(html_path + fn):
        os.makedirs(html_path + fn)
    if not os.path.exists(png_path + tp):
        os.makedirs(png_path + tp)
    if not os.path.exists(png_path + fn):
        os.makedirs(png_path + fn)

    # load the trained model
    model = torch.load(model_path)
    model.cpu()

    # wrap the model, such that it can be used for the lime explainer
    model_wrapped = LimeWrapper(model)
    tokenizer_wrapped = LimeTokenizer(model.embedding_module.tokenizer)
    lime_explainer = LimeTextExplainer(class_names=class_mapping)

    # load the data
    data_list, _, id_list, rational_list = load_data(data_path, targets, fold, label)

    # compute predictions
    true_positives, false_negatives, _, _ = predict_data(data_list, id_list, model, inv_class_mapping, label)

    # compute the explanations and neccessary properties for visualizations
    word_batch_list, weight_batch_list, probabilities, data_ids, exp_list = compute_explainations(true_positives, tokenizer_wrapped,
                                                                                                  lime_explainer, model_wrapped)
    tp = 'true_positives/'
    # convert the html files to images
    for i, (exp, d) in enumerate(zip(exp_list, data_ids)):
        html_file = html_path + tp + str(d) + '.html'
        exp.save_to_file(html_file)
    html_to_png(html_path + tp, png_path + tp)

    # compute the explanations and neccessary properties for visualizations
    word_batch_list, weight_batch_list, probabilities, data_ids, exp_list = compute_explainations(false_negatives, tokenizer_wrapped,
                                                                                                  lime_explainer, model_wrapped)
    fn = 'false_negatives/'
    # convert the html files to images
    for i, (exp, d) in enumerate(zip(exp_list, data_ids)):
        html_file = html_path + fn + str(d) + '.html'
        exp.save_to_file(html_file)
    html_to_png(html_path + fn, png_path + fn)


def compute_embeddings(model, data: list) -> list:

    feature_list = []
    print('Computing Features...')
    with tqdm(total=len(data), file=sys.stdout) as pbar:
        for sample in data:
            pbar.update(1)
            features = model.encoder([sample])
            feature_list.append(features.detach().cpu())
        features = torch.cat(feature_list).numpy()
        return features


def fit_umag(model, data):
    features = compute_embeddings(model, data)
    reducer = umap.UMAP()
    mapper = reducer.fit(features)
    return mapper


def compute_confusion_matrix(model, data: list, labels: list, inv_class_mapping):

    predictions = torch.argmax(torch.cat([model([sample], logits=False).detach().cpu() for sample in data]), dim=1).numpy()
    labels = np.asarray([inv_class_mapping[label] for label in labels])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    cm = confusion_matrix(labels, predictions, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(inv_class_mapping.keys()))
    disp.plot()
    cm = confusion_matrix(labels, predictions, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(inv_class_mapping.keys()))
    disp.plot()
    cm = confusion_matrix(labels, predictions, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(inv_class_mapping.keys()))
    disp.plot()


if __name__ == '__main__':

    inv_class_mapping = {
        'normal': 0,
        'offensive': 1,
        'hatespeech': 2
    }

    class_mapping = {
        0: 'Normal',
        1: 'Offensive',
        2: 'Hatespeech'
    }

    label = ['hatespeech']
    source = 'Sexual Orientation'
    targets = 'Religion'
    fold = [4]
    data_path = 'data/hateXplain.tsv'
    data_list, _, id_list, rational_list = load_data(data_path, targets, fold, label)
    model_path = 'experiments/curriculum/1/0.2/' + source + '/' + targets + '/' + str(fold[0]) + '/f1_best.pt'

    # load the model
    model = torch.load(model_path)
    model.cpu()
    model_wrapped = LimeWrapper(model)
    tokenizer_wrapped = LimeTokenizer(model.embedding_module.tokenizer)
    lime_explainer = LimeTextExplainer(class_names=class_mapping)

    # # load the data
    compute_lime_explainations('visualization/curriculum/1/0.2/' + source + '/' + targets + '/' + label[0] + '/',
                               model_path, class_mapping, inv_class_mapping, label[0])

    adversarial_path = 'visualization/mixup/' + source + '/' + targets + '/' + label[0] + '/' + 'png/'
    curriculum_path = 'visualization/curriculum/' + source + '/' + targets + '/' + label[0] + '/' + 'png/'
    baseline_path = 'visualization/baseline/' + source + '/' + targets + '/' + label[0] + '/' + 'png/'
    mixup_path = 'visualization/mixup/' + source + '/' + targets + '/' + label[0] + '/' + 'png/'
    out_path = 'visualization/common/'

    png_paths = [baseline_path, adversarial_path, mixup_path, curriculum_path]
    name_list = ['baseline', 'adversarial', 'mixup', 'curriculum']

    # id_dict = find_common_explainations(png_paths)
    # for key, value in id_dict.items():
    #     print(key)
    #     for v in value:
    #         print(v)
    #     print()
    # store_common_explainations(id_dict, name_list, out_path)

    # data_id = 3181
    # data = data_list[id_list.index(data_id)]
    # print(data)
    # p = compute_probabilities(data, model)
    # print(p)
    #
    # label = ['normal', 'offensive', 'hatespeech']
    # source = 'Sexual Orientation'
    # target = 'Sexual Orientation'
    # fold_list = [0, 1, 2, 3, 4]
    # fold = 4

    data_path = 'data/hateXplain.tsv'
    # data_list_source, label_list_source, id_list_source, rational_list_source = load_data(data_path, source, fold_list, label)
    # data_list_target, label_list_target, id_list_target, rational_list_target = load_data(data_path, target, fold_list, label)
    # data_list = deepcopy(data_list_source)
    # data_list.extend(data_list_target)
    #
    # model_path = 'experiments/baseline/1/' + source + '/' + source + '/' + str(fold) + '/f1_best.pt'
    # model = torch.load(model_path)
    # model.cuda()

    # compute_confusion_matrix(model, data_list_source, label_list_source, inv_class_mapping)

    #
    # mapper = fit_umag(model, data_list)
    #
    # features_source = compute_embeddings(model, data_list_source)
    # features_target = compute_embeddings(model, data_list_target)
    # features_src = mapper.transform(features_source)
    # features_tar = mapper.transform(features_target)
    # print(features_src.shape, features_tar.shape)
    # color_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    #
    # plt.subplot(121)
    # for class_idx in range(len(label)):
    #     idx_list = [i for i, j in enumerate(label_list_source) if label.index(j) == class_idx]
    #     label_features = np.take(features_src, idx_list, axis=0)
    #     plt.plot(label_features[:, 0], label_features[:, 1], '.', c=color_list[class_idx], markersize=3)
    # plt.subplot(122)
    # for class_idx in range(len(label)):
    #     idx_list = [i for i, j in enumerate(label_list_target) if label.index(j) == class_idx]
    #     label_features = np.take(features_target, idx_list, axis=0)
    #     plt.plot(label_features[:, 0], label_features[:, 1], '.', c=color_list[class_idx], markersize=3)
    # plt.show()
