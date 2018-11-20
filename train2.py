#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 02:43:32 2018

@author: mrudul
"""

import argparse
import logging
import random
import time
import numpy as np
import glob
import os

import torch
import torch.nn as nn
from torch import optim

import AMR
from models import AttnDecoderRNN, EncoderRNN, ChildSum

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda")

SOS_token = "<SOS>"
EOS_token = "<EOS>"
OOV_token = "<OOV>"

SOS_index = 0
EOS_index = 1
OOV_index = 2
MAX_LENGTH = 30
DROPOUT_P = 0.1



class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_words(self, words):
        for word in words:
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class AMRVocab:
    """ This class handles the mapping between the AMR nodes, words, & edges and their indicies
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token, OOV_index: OOV_token}
        self.edge2index = {None: 0}
        self.edge2count = {}
        self.index2edge = {0: None}
        self.n_words = 3
        self.n_edges = 1

    def add_graph(self, node):
        node_str = node.inst
        node_strs = node_str.split('-')
        try:
            d = int(node_strs[-1])
            node_str = '-'.join(node_strs[:-1])
        except ValueError:
            pass
        self._add_node(node_str)
        for edge,c_node in node.child.items():
            if type(c_node) is str or type(c_node) is float:
                self._add_edge(edge)
                self._add_node(c_node)
            else:
                self.add_graph(c_node)

    def _add_node(self, node):
        if node not in self.word2index:
            self.word2index[node] = self.n_words
            self.word2count[node] = 1
            self.index2word[self.n_words] = node
            self.n_nodes += 1
        else:
            self.word2count[node] += 1
    
    def _add_edge(self, edge):
        if edge not in self.edge2index:
            self.edge2index[edge] = self.n_edges
            self.edge2count[edge] = 1
            self.index2edge[self.n_edges] = edge
            self.n_edges += 1
        else:
            self.edge2count[edge] += 1
            
    def prune(self):
        for node in self.word2count:
            if self.word2count[node] < 2:
                ind = self.word2index[node]
                del self.word2index[node]
                self.index2word[ind] = OOV_token
        for edge in self.edge2count:
            if self.edge2count[edge] < 2:
                ind = self.edge2index[edge]
                del self.edge2index[edge]
                self.index2edge[ind] = None


######################################################################

def make_vocabs(train_pairs):
    """ Creates the vocabs for AMR and English based on the training corpus.
    """
    amr_vocab = AMRVocab()
    eng_vocab = Vocab()

    for snt,amr in train_pairs:
        eng_vocab.add_words(snt)
        amr_vocab.add_graph(amr.root)

    logging.info('amr (src) vocab size: %s, edge_vocab_size: %s', amr_vocab.n_nodes, amr_vocab.n_edges)
    logging.info('eng (tgt) vocab size: %s', eng_vocab.n_words)

    return amr_vocab, eng_vocab

######################################################################

def tensor_from_sentence(vocab, sentence, max_length = None):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence:
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    sen_len = len(indexes)
    if max_length is None:
        max_length = sen_len + 1
    indexes += [EOS_index] * (max_length - sen_len)
    #for i in range(sen_len,max_length):
    #    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_batch(vocab, pairs):
    """creates a tensor from a batch of amr-sentence pairs
    """
    snt_max = len(max(pairs, key=lambda x: len(x[0]))[0]) + 1
    pair = pairs[0]
    snt_tensor = tensor_from_sentence(vocab, pair[0], snt_max)
    for i in range(1,len(pairs)):
        pair = pairs[i]
        snt_tensor = torch.cat((snt_tensor,tensor_from_sentence(vocab, pair[0], snt_max)), 1)
    return snt_tensor

######################################################################
    
def encode_node(node, amr_vocab, encoder, child_sum, processed, outputs):
    if node.id in processed:
        return processed[node.id]
    ind = amr_vocab.node2index[None]
    if node.inst in amr_vocab.node2index:
        ind = amr_vocab.node2index[node.inst]
    inp_node = torch.tensor([[ind]], device=device)
    edges = None
    hidden_states0 = None
    hidden_states1 = None
    for edge,child_node in node.child.items():
        edge_index = amr_vocab.edge2index[None]
        if edge in amr_vocab.edge2index:
            edge_index = amr_vocab.edge2index[edge]
        edge_tensor = torch.tensor([[edge_index]], device=device)
        hidden = None
        #leaf nodes
        if type(child_node) is str or type(child_node) is float:
            inp_child = amr_vocab.node2index[None]
            if child_node in amr_vocab.node2index:
                inp_child = amr_vocab.node2index[child_node]
            inp = torch.tensor([[inp_child]], device=device)
            outp, hidden = encoder(inp, None)
            outputs[0] = torch.cat((outputs[0], outp))
        #amr nodes
        else:
            outp, hidden = encode_node(child_node, amr_vocab, encoder, child_sum, processed, outputs)
        
        if edges is None:
            edges = edge_tensor
            hidden_states0 = hidden[0]
            hidden_states1 = hidden[1]
        else:
            edges = torch.cat((edges, edge_tensor))
            hidden_states0 = torch.cat((hidden_states0, hidden[0]))
            hidden_states1 = torch.cat((hidden_states1, hidden[1]))
    
    hidden_sum = None
    if edges is not None:
        hidden_sum = child_sum(edges, (hidden_states0, hidden_states1))
    output, hidden_final = encoder(inp_node, hidden_sum)
    
    processed[node.id] = (output, hidden_final)
    outputs[0] = torch.cat((outputs[0], output))   
    return output, hidden_final
    
def encode_amr(graph, amr_vocab, encoder, child_sum, max_in_length):
    outputs = [torch.zeros(1, 1, encoder.hidden_size, device=device)]
    output, hidden_state = encode_node(graph.root, amr_vocab, encoder, child_sum, {}, outputs)
    en_outputs = outputs[0]
    pad = max_in_length - en_outputs.size()[0]
    for i in range(pad):
        en_outputs = torch.cat((en_outputs, torch.zeros(1, 1, encoder.hidden_size, device=device)))
    return en_outputs, hidden_state

def train(pairs, target_snt, amr_vocab, encoder, child_sum, decoder, optimizer, criterion, max_length=MAX_LENGTH):

    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()
    
    max_in_length = max(pairs, key=lambda x: x[1].size)[1].size + 1
    max_out_length = target_snt.size()[0]
    bsize = target_snt.size()[1]
    optimizer.zero_grad()
    #Encode
    encoder_outputs = None
    hidden0 = None
    hidden1 = None
    for snt,input_amr in pairs:
        en_out, hidden = encode_amr(input_amr, amr_vocab, encoder, child_sum, max_in_length)
        if encoder_outputs is None:
            encoder_outputs = en_out
            hidden0 = hidden[0]
            hidden1 = hidden[1]
        else:
            encoder_outputs = torch.cat((encoder_outputs, en_out), dim=1)
            hidden0 = torch.cat((hidden0, hidden[0]), dim=1)
            hidden1 = torch.cat((hidden1, hidden[1]), dim=1)
    encoder_hidden = (hidden0, hidden1)
    #decode
    cutoff = min(max_length, max_out_length)
    decoder_input = torch.tensor([[SOS_index] * bsize], device=device)
    decoder_hidden = decoder.get_initial_hidden_state(encoder_hidden)
    decoder_outputs = torch.zeros(max_length, bsize, decoder.output_size, device=device)
    for di in range(cutoff):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_outputs[di] += decoder_output[0]
        topv, topi = decoder_output.data.topk(1)
        decoder_input = target_snt[di:di+1]
        
    
    target = target_snt.transpose(0,1)
    output = decoder_outputs.transpose(2,1).transpose(0,2)
    loss = criterion(output[:,:,:cutoff], target[:,:cutoff])
    loss.backward()
    optimizer.step()
    return loss.item() 



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--edge_size', default=20, type=int,
                    help='embedding dimension of edges')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--train_files', default='../amr_anno_1.0/data/split/training/*',
                    help='training files.')
    ap.add_argument('--log_dir', default='./log',
                    help='log directory')
    ap.add_argument('--exp_name', default='experiment',
                    help='experiment name')
    ap.add_argument('--batch_size', default=5, type=int,
                    help='batch size')
    ap.add_argument('--load_checkpoint', action='store_true',
                    help='use existing checkpoint')

    args = ap.parse_args()
    
    logdir = args.log_dir
    exp_dir = logdir + '/' + args.exp_name
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    load_state_file = None
    if args.load_checkpoint:
        max_iter = 0
        state_files = glob.glob(exp_dir + '/*')
        for sf in state_files:
            iter_num = int(sf.split('_')[1].split('.')[0])
            if iter_num > max_iter:
                max_iter = iter_num
                load_state_file = sf
    # Create vocab from training data
    iter_num = 0
    train_files = glob.glob(args.train_files)
    train_pairs = AMR.read_AMR_files(train_files, True)
    amr_vocab, en_vocab = None, None
    state = None
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    edge_size = args.edge_size
    drop = DROPOUT_P
    mlength = MAX_LENGTH
    if load_state_file is not None:
        state = torch.load(load_state_file)
        iter_num = state['iter_num']
        amr_vocab = state['amr_vocab']
        en_vocab = state['en_vocab']
        hidden_size = state['hidden_size']
        edge_size = state['edge_size']
        drop = state['dropout']
        mlength = state['max_length']
        logging.info('loaded checkpoint %s', load_state_file)
    else:
        amr_vocab, en_vocab = make_vocabs(train_pairs)
    encoder = EncoderRNN(amr_vocab.n_nodes, hidden_size).to(device)
    child_sum = ChildSum(amr_vocab.n_edges, edge_size, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, en_vocab.n_words, dropout_p=drop, max_length=mlength).to(device)
    
    #load checkpoint
    if state is not None:
        encoder.load_state_dict(state['enc_state'])
        child_sum.load_state_dict(state['sum_state'])
        decoder.load_state_dict(state['dec_state'])

    # set up optimization/loss
    params = list(encoder.parameters()) + list(child_sum.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()
    
    #load checkpoint
    if state is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    while iter_num < args.n_iters:
        num_samples = batch_size
        remaining = args.checkpoint_every - (iter_num % args.checkpoint_every)
        remaining2 = args.print_every - (iter_num % args.print_every)
        if remaining < batch_size:
            num_samples = remaining
        elif remaining2 < batch_size:
            num_samples = remaining2
        iter_num += num_samples
        random_pairs = random.sample(train_pairs, num_samples)
        target_snt = tensors_from_batch(en_vocab, random_pairs)
        loss = train(random_pairs, target_snt, amr_vocab, encoder, child_sum, decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'sum_state': child_sum.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'amr_vocab': amr_vocab,
                     'en_vocab': en_vocab,
                     'hidden_size': hidden_size,
                     'edge_size': edge_size,
                     'dropout': drop,
                     'max_length': mlength
                     }
            filename = 'state_%010d.pt' % iter_num
            save_file = exp_dir + '/' + filename
            torch.save(state, save_file)
            logging.debug('wrote checkpoint to %s', save_file)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)


if __name__ == '__main__':
    main()