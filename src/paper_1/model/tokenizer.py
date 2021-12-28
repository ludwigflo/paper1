from tokenizers.trainers import WordPieceTrainer, BpeTrainer, UnigramTrainer, WordLevelTrainer
from tokenizers import pre_tokenizers, processors, models, Tokenizer
from typing import List, Tuple, Union, Dict, Iterator
from tokenizers.decoders import ByteLevel, WordPiece
from abc import ABC, abstractmethod


# noinspection PyPropertyAccess
def train_tokenizer(corpus: Iterator[str], pre_tokenizer_seq: Tuple[str] = ('whitespace', ), tokenizer_model: str = 'word_level',
                    post_processing_template: Union[Tuple[str, str], None] = None,
                    special_tokens: Union[None, List[str]] = None, **kwargs):
    """

    Parameters
    ----------
    corpus
    pre_tokenizer_seq
    tokenizer_model
    post_processing_template
    special_tokens

    Returns
    -------

    """

    # check if the required pre-tokenizers are available
    pre_tokens = {'byte_level': pre_tokenizers.ByteLevel,
                  'whitespace': pre_tokenizers.Whitespace,
                  'whitespace_split': pre_tokenizers.WhitespaceSplit,
                  'punctuation': pre_tokenizers.Punctuation,
                  'digits': pre_tokenizers.Digits}

    # initialize the pre-tokenizer Sequence
    pre_token_list = []
    for tokenizer in pre_tokenizer_seq:
        assert tokenizer in pre_tokens, 'Pre-tokenizer ' + tokenizer + ' is not available.'
        pre_token_list.append(pre_tokens[tokenizer]())

    # check if the tokenizer model is available
    model_types = {'word_level': (models.WordLevel, WordLevelTrainer),
                   'word_piece': (models.WordPiece, WordPieceTrainer, WordPiece),
                   'unigram': (models.Unigram, UnigramTrainer),
                   'bpe': (models.BPE, BpeTrainer, ByteLevel)
                   }
    assert tokenizer_model in model_types, 'Tokenizer ' + tokenizer_model + ' is not available.'

    # get the keyword arguments for the trainer and the main tokenizer
    tokenizer_kwargs = kwargs['main_tokenizer'] if 'main_tokenizer' in kwargs else {}
    trainer_kwargs = kwargs['trainer'] if 'trainer' in kwargs else {}

    # create the main tokenizer
    tokenizer = Tokenizer(model_types[tokenizer_model][0](**tokenizer_kwargs))
    if special_tokens is not None:
        trainer = model_types[tokenizer_model][1](special_tokens=special_tokens, **trainer_kwargs)
    else:
        trainer = model_types[tokenizer_model][1](**trainer_kwargs)

    # create a list of pre-tokenizers
    pre_tokenizer = pre_tokenizers.Sequence(pre_token_list)
    tokenizer.pre_tokenizer = pre_tokenizer

    if len(model_types[tokenizer_model]) == 3:
        tokenizer.decoder = model_types[tokenizer_model][2]()

    # train the tokenizer
    tokenizer.train_from_iterator(corpus, trainer)

    # create a post_processor, if a template is provided
    if post_processing_template is not None:
        single, pair = post_processing_template[0], post_processing_template[1]
        if special_tokens is not None:
            special_token_tuple = [(token, tokenizer.get_vocab()[token]) for token in special_tokens]
            post_processor = processors.TemplateProcessing(single=single, pair=pair, special_tokens=special_token_tuple)
            tokenizer.post_processor = post_processor
    return tokenizer


class TokenizerBase(ABC):

    @abstractmethod
    def tokenize(self, input_data: list, token_ids: bool = True):
        pass

    @abstractmethod
    def ids_to_data(self, token_ids: list) -> List[str]:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abstractmethod
    def vocab_dict(self) -> dict:
        pass

    @abstractmethod
    def enable_padding(self, pad_token: str) -> None:
        pass

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        pass


class WordPieceTokenizer(TokenizerBase):

    def __init__(self, hugging_face_tokenizer: Tokenizer):
        self.hugging_face_tokenizer = hugging_face_tokenizer

    @property
    def vocab_size(self) -> int:
        """
        Size of the vocabulary of the tokenizer.

        Returns
        -------
        size: Size of the vocabulary of the tokenizer.
        """

        size = self.hugging_face_tokenizer.get_vocab_size(with_added_tokens=True)
        return size

    @property
    def vocab_dict(self) -> dict:
        """
        Returns the vocabulary dictionary of the tokenizer.

        Returns
        -------
        vocabulary: Vocabulary dictionary of the tokenizer.
        """

        vocabulary = self.hugging_face_tokenizer.get_vocab(with_added_tokens=True)
        return vocabulary

    def tokenize(self, input_data: list, token_ids: bool = False) -> Union[list, tuple]:
        """
        Tokenizes a batch of input data points.

        Parameters
        ----------
        input_data: List of strings, which should be tokenized.
        token_ids: Whether to return tokens or token ids.

        Returns
        -------
        tokens: List of tokenized text data, which are represented as list of strings.
        ids: list of token ids, which are represented as list of integers.
        """

        # encode the data batch
        encodings = self.hugging_face_tokenizer.encode_batch(input_data)

        # extract and return the tokens (and optionally the corresponding token ids)
        tokens = [encoding.tokens for encoding in encodings]
        if token_ids:
            ids = [encoding.ids for encoding in encodings]
            return tokens, ids
        else:
            return tokens

    def token_to_id(self, token: str) -> int:
        """
        Determines the id corresponding to a token.

        Parameters
        ----------
        token: Token, for which its corresponding id should be returned.

        Returns
        -------
        token_id: Id, corresponding to the provided token.
        """

        token_id = self.hugging_face_tokenizer.token_to_id(token)
        return token_id

    def ids_to_data(self, token_ids: list) -> list:
        """
        Converts a batch of token ids to their corresponding tokens.

        Parameters
        ----------
        token_ids: List of tokenized data points, which are represented as a list of integers.

        Returns
        -------
        data: Decoded text data, represented as a list of strings.
        """

        data = self.hugging_face_tokenizer.decode(token_ids)
        return data

    def enable_padding(self, pad_token: str) -> None:
        """
        Switch the tokenizer into a mode, in which it pads a batch of input data to the longest sequence in the batch.

        Parameters
        ----------
        pad_token: Pad token, which is used to pad the input.
        """

        self.hugging_face_tokenizer.enable_padding(pad_token=pad_token, pad_id=self.hugging_face_tokenizer.token_to_id(pad_token))

