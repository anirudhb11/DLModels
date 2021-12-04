import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k  # German to English Dataset
from torchtext.data import Field, BucketIterator  # Preprocessing

import numpy as np
import spacy  # Tokenizer
import random

from torch.utils.tensorboard import SummaryWriter  # Print to tensorBoard
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

# German -> English translation
spacy_ger = spacy.load('de')  # Tokenizer for German
spacy_eng = spacy.load('en')


def tokenizer_ger(text):
    """

    :param text: sentence that needs to be tokenized. For ex: 'Hello my name is'
    :return: Tokenized sentence. For ex: ['Hello', 'my', 'name', 'is']
    """
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, val_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)  # Word must appear >=2 times to add to the dataset
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p_drop):
        """
        :param input_size: vocabulary size
        :param embedding_size:
        :param hidden_size: size of vectors for LSTM layers
        :param num_layers:
        :param p_drop: Dropout probability
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p_drop)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p_drop)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        """
        :param self:
        :param x: Vector of shape (seq_len, batch_size)
        :return: Hidden and cell state
        """
        embedding = self.dropout(self.embedding(x))  # Shape: (seq_len, bs, embedding_size)
        encoder_states, (hidden, cell) = self.rnn(embedding)
        h_reshaped = torch.cat((hidden[0:1], hidden[1:2]), dim=2)  # Shape: (1, bs, 2 * hidden_sz)
        cell_reshaped = torch.cat((cell[0:1], cell[1:2]), dim=2)  # Shape: (1, bs, 2 * hidden_sz)
        hidden = self.fc_hidden(h_reshaped)
        cell = self.fc_cell(cell_reshaped)
        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p_drop):
        """

        :param input_size:
        :param embedding_size:
        :param hidden_size: For LSTM model, same as that of encoder
        :param output_size: target vocabulary size
        :param num_layers:
        :param p_drop:
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p_drop)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers, dropout=p_drop)

        self.energy = nn.Linear(3 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_states):
        # Shape of x is (bs)  we want it to be (1, bs), Given the previous hidden and cell state and the previous
        # word predicted the. So we have bs number of examples of a single word, this is in contrast with the
        # encoder where we have seq_len number of words of a batch being passed on as input
        # hidden shape: (num_layers, bs, hidden)
        # cell shape: (num_layers, bs, hidden)
        # encoder_states shape: (seq_len, bs, 2 * hidden)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))  # Shape (1, bs, emb_size)
        seq_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(seq_length, 1, 1)  # Shape (seq_len, bs, hidden)
        energies = self.relu(self.energy(torch.cat((encoder_states, h_reshaped), dim=2)))  # Shape (seq_len, bs, 1)
        attention = self.softmax(energies)
        attention = attention.permute(1, 2, 0)  # Shape (bs, 1, seq_len)
        encoder_states = encoder_states.permute(1, 0, 2)  # Shape (bs, seq_len, 2 * hidden)

        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)  # Shape (1, bs, 2 * hidden)
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # Output shape: (1, bs, hidden_sz)
        predictions = self.fc(outputs)  # Shape: (1, bs, vocab_sz)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        """

        :param source: sentence from source language, i.e German here,
        :param target: actual translated sentence, i.e English here
        :param teacher_force_ratio: probability indicating when you want to use the correct output word in decoder
        :return:
        """
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        encoder_states, hidden, cell = self.encoder(source)
        # Grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_states)
            outputs[t] = output

            best_guess = output.argmax(1)  # Shape: (bs)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


# Training hyper-params
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyper-params
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/Loss_plot')
step = 0

# Within a batch we want to have examples of similar length, minimize the number of padding
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, val_data, test_data),
                                                                      batch_size=batch_size, sort_within_batch=True,
                                                                      sort_key=lambda x: len(x.src),
                                                                      device=device)
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(
    device)
model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

for epoch in range(num_epochs):
    ctr = 0
    agg_loss = 0
    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(input_data, target)  # (targ_len, bs, targ_vocab_sz)

        # Cross entropy expects the following:
        # predictions are of shape (bs, vocab_size) and targets are shape bs
        output = output[1:].reshape(-1, output.shape[2])  # First output is start token
        target = target[1:].reshape(-1)  # Shape: (targ_len, bs)

        optimizer.zero_grad()
        loss = criterion(output, target)
        agg_loss += loss
        ctr += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1
    print(f'Epoch [{epoch} / {num_epochs}] Loss: {agg_loss/ctr}')

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)
