#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 23:15:34 2018

@author: mrudul
"""


from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn

device = torch.device("cuda")

#the class for attention model
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
        self.V_a = nn.Linear(hidden_size, 1)
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.U_a = nn.Linear(hidden_size*2, hidden_size)
        
    def forward(self, input, encoder_outputs):
        intermediate = self.W_a(input) + self.U_a(encoder_outputs)
        _e = self.V_a(torch.tanh(intermediate))
        weights = torch.nn.Softmax(dim=0)(_e)
        return torch.sum(weights * encoder_outputs, dim=0), weights
    
#the class for child-sum hidden states
class ChildSum(nn.Module):
    def __init__(self, edge_size, edge_embedding_size, hidden_size):
        super(ChildSum, self).__init__()
        self.edge_size = edge_size
        self.hidden_size = hidden_size
        self.edge_embedding = nn.Embedding(edge_size, edge_embedding_size)
        self.U = nn.Linear(edge_embedding_size + hidden_size, hidden_size)
        self.W = nn.Linear(edge_embedding_size + hidden_size, hidden_size)
    
    def forward(self, edges, hidden_states):
        edge_states = self.edge_embedding(edges)
        h0,h1 = hidden_states
        hidden_all = self.U(torch.cat((h0, edge_states), dim=2))
        cell_all = self.W(torch.cat((h0, edge_states), dim=2))
        hidden = torch.sum(hidden_all, dim=0).view(1,1,-1)
        cell = torch.sum(cell_all, dim=0).view(1,1,-1)
        return hidden, cell


class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size)


    def forward(self, input, hidden):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        inp = self.embedding(input)
        output, hidden = self.encoder(inp, hidden)     
        return output, hidden

    def get_initial_hidden_state(self, bsize = 1):
        return torch.zeros(1,bsize,self.hidden_size,device=device)


class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=30):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.decoder = nn.LSTM(hidden_size*3, hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attention = Attention(hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        inp = self.embedding(input)
        #dropout
        if self.training:
            inp = self.dropout(inp)
        #get context vector
        context, attn_weights = self.attention(inp, encoder_outputs)
        bsize = context.size()[0]
        context = context.view(1,bsize,-1)
        _inp = torch.cat((context, inp), 2)
        output, hidden = self.decoder(_inp, hidden)
        _out = self.out(output)
        log_softmax = nn.LogSoftmax(dim=2)(_out)
        return log_softmax, hidden, attn_weights.view(-1,bsize)

    def get_initial_hidden_state(self, encoder_hidden):
        return encoder_hidden