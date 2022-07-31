# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for text data."""
import string
import numpy as np
import torch


class SimpleVocab(object):

  def __init__(self):
    super(SimpleVocab, self).__init__()
    self.word2id = {}
    self.wordcount = {}
    self.word2id['<UNK>'] = 0
    self.wordcount['<UNK>'] = 9e9

  def tokenize_text(self, text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    # ex: make bottom-left large circle cyan
    tokens = str(text).lower()
    tokens = tokens.translate(str.maketrans('','',string.punctuation))
    # ex: make bottomleft large circle cyan
    tokens = tokens.strip().split()
    # ex: ['make', 'bottomleft', 'large', 'circle', 'cyan']
    return tokens

  def build(self, texts):
    for text in texts:
      # ex: make bottom-left large circle cyan
      tokens = self.tokenize_text(text)
      for token in tokens:
        if token not in self.wordcount:
          self.wordcount[token] = 0
        self.wordcount[token] += 1
    # ex: wordcount:{'<UNK>': 9000000000.0, 'make': 2, 'bottomleft': 2, 'large': 2, 'circle': 1, 'cyan': 1, 'purple': 1, 'object': 1, 'blue': 1}
    for token in sorted(list(self.wordcount.keys())):
      if token not in self.word2id:
        self.word2id[token] = len(self.word2id)
    # ex: word2id:{'<UNK>': 0, 'blue': 1, 'bottomleft': 2, 'circle': 3, 'cyan': 4, 'large': 5, 'make': 6, 'object': 7, 'purple': 8}

  def threshold_rare_words(self, wordcount_threshold=5):
    for w in self.word2id:
      if self.wordcount[w] < wordcount_threshold:
        self.word2id[w] = 0

  def encode_text(self, text):
    tokens = self.tokenize_text(text)
    # ->ex: ['make', 'bottomleft', 'large', 'circle', 'cyan']
    x = [self.word2id.get(t, 0) for t in tokens]
    # ->ex:[6, 2, 5, 3, 4]
    return x

  def get_size(self):
    return len(self.word2id)


class TextLSTMModel(torch.nn.Module):

  def __init__(self,
               texts_to_build_vocab,
               word_embed_dim=512,
               lstm_hidden_dim=512):

    super(TextLSTMModel, self).__init__()

    self.vocab = SimpleVocab()
    self.vocab.build(texts_to_build_vocab)
    vocab_size = self.vocab.get_size()

    self.word_embed_dim = word_embed_dim
    self.lstm_hidden_dim = lstm_hidden_dim
    self.embedding_layer = torch.nn.Embedding(vocab_size, word_embed_dim)
    self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim)
    self.fc_output = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
    )

  def forward(self, x):
    """ input x: list of strings"""
    # print("TextLSTMModel x List:")
    # print(len(x))
    # 32
    # print(x) ->['make bottom-left large circle cyan', 'make bottom-left large purple object blue',...
    if type(x) is list:
      if type(x[0]) is str or type(x[0]) is unicode:
        # convert to index list of text->ex:[6, 2, 5, 3, 4]
        x = [self.vocab.encode_text(text) for text in x]

    assert type(x) is list
    assert type(x[0]) is list
    assert type(x[0][0]) is int
    # print("TextLSTMModel x List 2:")
    # print(len(x))
    # 32
    # print(x)-->ex: [[20, 14, 11, 19, 16], [12, 14, 11, 19, 26, 21],...
    return self.forward_encoded_texts(x)

  def forward_encoded_texts(self, texts):
    # print("forward_encoded_texts texts:")
    # print(texts)-->ex: [[20, 14, 11, 19, 16], [12, 14, 11, 19, 26, 21],...
    # to tensor
    # to get lengths of index list of each text-->ex: [5, 6,...
    lengths = [len(t) for t in texts]
    # create 0 matrix with [max length within texts, numbers of texts], ex: [10, 32]
    itexts = torch.zeros((np.max(lengths), len(texts))).long()
    """each column presents each index numbers of each text, there are batch size columns:
       itexts ex:
       tensor([[20, 12,...],
        [14, 14,...],
        [11, 11,...],
        [19, 19,...],
        [16, 26,...],
        [ 0, 21,...]])
    """
    for i in range(len(texts)):
      itexts[:lengths[i], i] = torch.tensor(texts[i])

    # embed words
    itexts = torch.autograd.Variable(itexts).cuda()
    # print("TextLSTMModel itexts shape:")
    # print(itexts.shape)
    # torch.Size([6, 32])
    etexts = self.embedding_layer(itexts)
    # print("TextLSTMModel etexts shape:")
    # print(etexts.shape)
    # torch.Size([6, 32, 512])
    # lstm
    lstm_output, _ = self.forward_lstm_(etexts)
    # -> ex: lstm_output shape: torch.Size([6, 32, 512])
    '''
    tensor([[[ 0.0189,  0.0054, -0.0153,  ..., -0.0207,  0.0243, -0.0199],
         [ 0.0189,  0.0054, -0.0153,  ..., -0.0207,  0.0243, -0.0199]],...
    '''
    # get last output (using length)
    text_features = []
    for i in range(len(texts)):
      '''
      get final text features via one by one sequence, the final text feature is the last in current sequence:
      "lengths[i] - 1": in the the 1st dimension(seq): the actual length of this sequence.
      "i": in the 2nd dimension(batch): which sequence in this batch.
      ":": get all in the 3rd dimension(features)
      '''
      text_features.append(lstm_output[lengths[i] - 1, i, :])

    # output
    # print("TextLSTMModel text_features List2:")
    # print(len(text_features))
    # 32
    # print(text_features)-->List contains actual each text features, shape:torch.Size([512])
    # convert text_features List to Tensor
    text_features = torch.stack(text_features)
    text_features = self.fc_output(text_features)
    # print("TextLSTMModel text_features shape3:")
    # print(text_features.shape)
    # torch.Size([32, 512])
    # each row is 512 embedded features values
    '''
    tensor([[-0.0007,  0.0070,  0.0006,  ...,  0.0371, -0.0068,  0.0143],
        [-0.0002,  0.0291, -0.0598,  ...,  0.0240,  0.0274,  0.0371]],
        ...
       grad_fn=<AddmmBackward0>)
    '''
    return text_features

  def forward_lstm_(self, etexts):
    # ex: etexts shape: torch.Size([6, 32, 512])
    # etexts.shape[1]:32 is batch size
    batch_size = etexts.shape[1]
    # first_hidden is a tuple:(h_0, c_0)
    # because num_layer is 1
    # so:
    # h_0 shape:torch.Size([1*1, 32, 512])
    # c_0 shape:torch.Size([1*1, 32, 512])
    first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
                    torch.zeros(1, batch_size, self.lstm_hidden_dim))
    first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
    lstm_output, last_hidden = self.lstm(etexts, first_hidden)
    return lstm_output, last_hidden
