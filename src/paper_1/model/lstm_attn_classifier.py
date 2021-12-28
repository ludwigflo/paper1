from paper_1.model.tokenizer import train_tokenizer, WordPieceTokenizer
from paper_1.utils import read_parameter_file
from paper_1.model.embeddings import EmbeddingModule
from typing import List, Union
from torch import nn
import torch


class LSTMAttnClassifier(nn.Module):
    def __init__(self, num_classes: int, attention_dim: int, num_attention_heads: int,
                 lstm_kwargs: Union[dict, None] = None, lstm: Union[nn.Module, None] = None,
                 embedding_module: Union[EmbeddingModule, None] = None, embedding_args: Union[tuple, None] = None):
        """
        LSTM Classification model with an self attentive sentence embedding module. Proposed in
        https://arxiv.org/pdf/1703.03130.pdf.

        Parameters
        ----------
        num_classes: Number of classes in the classification task.
        attention_dim: TODO: Documentation
        num_attention_heads: TODO: Documentation
        lstm_kwargs: (Optional) Key word arguments, required for initializing a Pytorch lstm model (required, if no such model is provided in the constructor).
        lstm: (Optional) Pretrained lstm model (required, if no lstm_kwargs are provided).
        embedding_module: (Optional) Module which represents embeddings (required, if no embedding_args are provided).
        embedding_args: (Optional) (Tokenizer, embedding_dim), required for initializing an EmbeddingModule (only required, if no EmbeddingModule is provided).
        """

        super().__init__()

        # check if all the required information is provided
        assert embedding_module is not None or embedding_args, "Need to provide either an embedding module or arguments, " \
                                                               "which are required for initializing such a module"
        assert lstm is not None or lstm_kwargs, "Need to provide either a lstm model or arguments, " \
                                                "which are required for initializing such a model"

        # initialize an embedding module, if not such module hasn't been provided
        if embedding_module is None:
            self.embedding_module = EmbeddingModule(*embedding_args)

        # otherwise, store the provided embedding module
        else:
            self.embedding_module = embedding_module
        self.embedding_module.tokenizer.enable_padding('<pad>')

        # get the size of the embeddings
        embedding_dim = self.embedding_module.embedding_dim

        # initialize a LSTM feature extractor, if no such module has been provided
        if lstm is None:
            self.lstm = nn.LSTM(input_size=embedding_dim, batch_first=True, **lstm_kwargs)
        else:
            self.lstm = lstm

        # initialize a dropout layer
        self.dropout = nn.Dropout(self.lstm.dropout)
        num_directions = 2 if self.lstm.bidirectional else 1

        # initialize the attention module
        self.attention_module = nn.Sequential(

            # N x Seq_dim x H*L*S
            nn.Linear(self.lstm.hidden_size * num_directions, attention_dim),
            nn.Tanh(),

            # N x Seq_dim x attention_dim
            nn.Linear(attention_dim, num_attention_heads),

            # N x Seq_dim x num_attention_heads
            nn.Softmax(dim=1)
        )

        # linear output Layer
        self.output_layer = nn.Linear(self.lstm.hidden_size * num_directions * num_attention_heads, num_classes)

        # softmax non-linearity
        self.softmax = nn.Softmax(dim=1)

    def compute_lstm_state(self, data_batch: Union[List[str], List[List[str]]], tokenize: bool = True) -> torch.Tensor:
        """

        Returns
        -------

        """

        # embeddings with dimensions (batch x seq_len x feature_dim)
        embeddings = self.embedding_module(data_batch, return_tensor=True, tokenize=tokenize)

        # forward the embeddings through the LSTM
        final_hidden_state, (_, _) = self.lstm(embeddings)
        return final_hidden_state

    def encoder(self, data_batch: Union[List[str], List[List[str]]], tokenize: bool = True) -> torch.Tensor:
        """
        Encoder of the LSTMClassifier model.

        Parameters
        ----------
        data_batch: Batch of input text data, represented as a list of strings.
        tokenize: Whether to tokenize the input data or not.

        Returns
        -------
        encodings: Feature representation of the data batch.
        """

        final_hidden_state = self.compute_lstm_state(data_batch,tokenize)

        # apply dropout regularization to the final hidden states, which we aim to use as features
        encodings = self.dropout(final_hidden_state)

        # use the attention module for computing the final feature representation
        features = self.apply_attention(encodings)
        features = features.view(features.size()[0], -1)
        return features

    def compute_attention_weights(self, encodings: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        encodings

        Returns
        -------

        """

        attention_weights = self.attention_module(encodings)
        return attention_weights

    def apply_attention(self, encodings: torch.Tensor):
        """

        Parameters
        ----------
        encodings

        Returns
        -------

        """

        attention_weights = self.compute_attention_weights(encodings)
        feature_map = torch.einsum('bsd,bsr->bdr', encodings, attention_weights)
        return feature_map

    def compute_average_attention(self, encodings: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        encodings

        Returns
        -------

        """

        attention_weights = self.compute_attention_weights(encodings)
        avg_attention_weights = torch.mean(attention_weights, dim=2)
        return avg_attention_weights

    def forward(self, data_batch: List[str], tokenize: bool = True, logits: bool = True) -> torch.Tensor:
        """
        Forward pass of the LSTMClassifier.

        Parameters
        ----------
        data_batch: Batch of input text data, represented as a list of strings.
        tokenize: Whether to tokenize the input data or not.
        logits: Whether the model returns logits or probabilities.

        Returns
        -------
        output: Classification predictions (logits or probabilities), created by the model.
        """

        # compute the feature representation fof the input
        features = self.encoder(data_batch, tokenize)

        # compute the logits
        output = self.output_layer(features)

        # compute probabilities, if required
        if not logits:
            output = self.softmax(output)
        return output
