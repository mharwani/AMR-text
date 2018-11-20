#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 05:52:06 2018

@author: mrudul
"""


import argparse
import logging
import random
import time
import glob
import os
from nltk.translate.bleu_score import corpus_bleu

import torch

import AMR
from models import AttnDecoderRNN, EncoderRNN, ChildSum
from train import encode_amr, SOS_index, EOS_index, EOS_token, AMRVocab

device = torch.device("cuda")



def translate(encoder, child_sum, decoder, input_amr, amr_vocab, max_length):
    """
    runs tranlsation, returns the output
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        #encode
        encoder_outputs, encoder_hidden = encode_amr(input_amr, amr_vocab, encoder, child_sum, input_amr.size + 1)
        #decode
        cutoff = max_length
        decoder_input = torch.tensor([[SOS_index]], device=device)
        decoder_hidden = decoder.get_initial_hidden_state(encoder_hidden)
        decoder_outputs = torch.zeros(max_length, 1, decoder.output_size, device=device)
        decoded_words = []
        for di in range(cutoff):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[di] += decoder_output[0]
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                break
            else:
                decoded_words.append(amr_vocab.index2word[topi.item()])

            decoder_input = topi.view(1,-1)
        
        return decoded_words


######################################################################

# Translate (dev/test)set takes in a list of amrs and writes out their transaltes
def translate_amrs(encoder, child_sum, decoder, pairs, amr_vocab, max_length, max_num_sentences=None):
    output_sentences = []
    for snt,amr in pairs[:max_num_sentences]:
        output_words = translate(encoder, child_sum, decoder, amr, amr_vocab, max_length)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_amr(encoder, child_sum, decoder, pairs, amr_vocab, max_length, n=1):
    for i in range(n):
        snt,amr = random.choice(pairs)
        print(amr.print())
        print('=', ' '.join(snt))
        output_words = translate(encoder, child_sum, decoder, amr, amr_vocab, max_length)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        references = [snt]
        candidates = [output_sentence.split()]
        dev_bleu = corpus_bleu([references], candidates)
        logging.info('Sentence BLEU score: %.2f', dev_bleu)
        print('')


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev_files', default='../amr_anno_1.0/data/split/test/*',
                    help='dev files.')
    ap.add_argument('--log_dir', default='./log',
                    help='log directory')
    ap.add_argument('--exp_name', default='experiment',
                    help='experiment name')
    args = ap.parse_args()
    
    #read dev files
    dev_files = glob.glob(args.dev_files)
    dev_pairs = AMR.read_AMR_files(dev_files, True)
    
    logdir = args.log_dir
    exp_dir = logdir + '/' + args.exp_name
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    max_iter = 0
    dev_bleu = 0.0
    while True:
        load_state_file = None
        state_files = glob.glob(exp_dir + '/*')
        for sf in state_files:
            iter_num = int(sf.split('_')[1].split('.')[0])
            if iter_num > max_iter:
                max_iter = iter_num
                load_state_file = sf
        if load_state_file is not None:
            state = torch.load(load_state_file)
            amr_vocab = state['amr_vocab']
            hidden_size = state['hidden_size']
            edge_size = state['edge_size']
            drop = state['dropout']
            mlength = state['max_length']
            logging.info('loaded checkpoint %s', load_state_file)
            
            encoder = EncoderRNN(amr_vocab.n_words, hidden_size).to(device)
            child_sum = ChildSum(amr_vocab.n_edges, edge_size, hidden_size).to(device)
            decoder = AttnDecoderRNN(hidden_size, amr_vocab.n_words, dropout_p=drop, max_length=mlength).to(device)
            encoder.load_state_dict(state['enc_state'])
            child_sum.load_state_dict(state['sum_state'])
            decoder.load_state_dict(state['dec_state'])
            # translate from the dev set
            translate_random_amr(encoder, child_sum, decoder, dev_pairs, amr_vocab, mlength, n=10)
            translated_amrs = translate_amrs(encoder, child_sum, decoder, dev_pairs, amr_vocab, mlength)
            references = [[pair[0]] for pair in dev_pairs[:len(translated_amrs)]]
            candidates = [sent.split() for sent in translated_amrs]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)
        else:
            logging.info('No new checkpoint found. Last DEV BLEU score: %.2f', dev_bleu)
        
        time.sleep(20)

if __name__ == '__main__':
    main()