from paper_1.model.tokenizer import train_tokenizer, TokenizerBase, WordPieceTokenizer
from typing import Union, List
import torch.nn as nn
import pickle
import torch


class EmbeddingModule(nn.Module):

    def __init__(self, tokenizer: TokenizerBase, embedding_dim: int = 64, pad_token: str = '<pad>', unk_token: str = '<unk>'):

        # call the constructor of the super class
        super().__init__()

        # store the unk_token
        self.unk_token = unk_token
        self.unk_token_id = tokenizer.token_to_id(unk_token)

        # store the tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding(pad_token=pad_token)

        # initialize the embeddings
        self.embedding_dim = embedding_dim
        self.vocab_size: int = tokenizer.vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, embedding_dim)

    def set_embedding_weights(self, weights: torch.Tensor) -> None:
        """
        Sets the parameters of self.embeddings to pre-initialized embedding weights.

        Parameters
        ----------
        weights: Weights, which are used to initialize the embedding weights.
        """

        # get the size of self.embeddings
        embedding_size = self.embeddings.weight.size()

        # get the size of the provided weights
        weight_size = weights.size()

        # check if the sizes match
        assert weight_size == embedding_size, "Size of the provided embedding weights needs to be " \
                                              "equal to the embedding size of this module."

        # assign the weights
        self.embeddings.weight.data.copy_(weights)

    def get_vocab(self) -> dict:
        """
        Returns the vocabulary of the tokenizer object.

        Returns
        -------
        vocab: Vocabulary of the tokenizer object.
        """

        vocab = self.tokenizer.vocab_dict
        return vocab

    def forward(self, input_data: Union[List[str], List[List[int]]], tokenize: bool = True,
                return_tensor: bool = True) -> Union[list, torch.Tensor]:
        """
        Forward pass of the embedding object.

        Parameters
        ----------
        input_data: Data, for which the embedding sequence should be returned.
        tokenize: Whether to tokenize the data, or the data is already tokenized.
        return_tensor: Whether to convert the embedding list to a Pytorch Tensor (only, if the data points are padded to equal length).

        Returns
        -------
        embeddings: Embeddings, corresponding to the provided input data.
        """

        # tokenize the data, if required
        if tokenize:
            _, id_list = self.tokenizer.tokenize(input_data, token_ids=True)
        else:
            id_list = []
            for data in input_data:
                id_list.append([self.tokenizer.token_to_id(token) for token in data])

        # remove None values
        for i, id_list_act in enumerate(id_list):
            id_list[i] = [x if x is not None else self.unk_token_id for x in id_list_act]

        id_list = [torch.Tensor(ids).long() for ids in id_list]

        # use the gpu, if possible
        if next(self.parameters()).is_cuda:
            id_list = [x.cuda() for x in id_list]

        # get the embeddings as list
        embedding_list = [self.embeddings(ids) for ids in id_list]

        # return a pytorch tensor, if required
        if return_tensor:
            embeddings = torch.stack(embedding_list)
            return embeddings

        # Otherwise, return a list of pytorch tensors
        else:
            embeddings = embedding_list
            return embeddings
