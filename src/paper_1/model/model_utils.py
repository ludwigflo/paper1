from paper_1.model.embeddings import EmbeddingModule
from paper_1.model.lstm_attn_classifier import LSTMAttnClassifier
from paper_1.utils import read_parameter_file
import torch.nn as nn
import torch
import pickle


def initialize_model(model_params: dict, base_dir: str, device: str) -> nn.Module:
    """
    initializes a model, which is used as a baseline during our experiments.

    Parameters
    ----------
    model_params: Dictionary, in which all necessary model parameters are stored.
    base_dir: Base directory of the repository.
    device:

    Returns
    -------
    model: Model, which is initialized according to the model parameters.
    """

    # extract the parameters
    embedding_path = base_dir + model_params['embedding_path']

    # load the pretrained embedding module
    embedding_module = torch.load(embedding_path)

    # arguments, which are used to create a lstm model
    model_params = model_params[model_params['model_type']]

    # create the model
    model = LSTMAttnClassifier(embedding_module=embedding_module, **model_params)

    # use the specified device
    model.to(device)
    return model


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "tokenizer":
            renamed_module = "paper_1.model.tokenizer"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


if __name__ == '__main__':
    #
    # with open('tokenizer.pickle', 'rb') as f:
    #     tokenizer = pickle.load(f)
    #     print(tokenizer)
    #
    # embedding_module = EmbeddingModule(tokenizer, embedding_dim=300)
    # weights = torch.load('embeddings.pt').weight.data
    # embedding_module.set_embedding_weights(weights)
    # torch.save(embedding_module, 'embedding_module.pt')
    #
    embedding_module = torch.load('embedding_module.pt')
    print(embedding_module)
