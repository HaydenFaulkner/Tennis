"""
Transformer
=================================

This example shows how to implement the Transformer model with Gluon NLP Toolkit.

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones,
          Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6000--6010},
  year={2017}
}
"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import time
import random
import os
import logging
import math
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

from gluonnlp.loss import MaskedSoftmaxCELoss, LabelSmoothing
from gluonnlp.model.translation import NMTModel
from gluonnlp.model.transformer import get_transformer_encoder_decoder, ParallelTransformer
from gluonnlp.utils.parallel import Parallel
from utils.translation import BeamSearchTranslator

from metrics.bleu import _bpe_to_words, compute_bleu
from utils import captioning
from dataset import TennisSet
from models.vision.definitions import FrameModel
from utils.layers import TimeDistributed
from gluoncv.model_zoo import get_model
from mxnet.gluon.data.vision import transforms

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Transformer Model')

parser.add_argument('--model_id', type=str, default='0000', help='model identification string')
parser.add_argument('--src_lang', type=str, default='en', help='Source language')
parser.add_argument('--tgt_lang', type=str, default='de', help='Target language')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--num_units', type=int, default=4096, help='Dimension of the embedding '
                                                                'vectors and states.')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='Dimension of the hidden state in position-wise feed-forward networks.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='epsilon parameter for label smoothing')
parser.add_argument('--num_layers', type=int, default=6,
                    help='number of layers in the encoder and decoder')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads in multi-head attention')
parser.add_argument('--scaled', action='store_true', help='Turn on to use scale in attention')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size. Number of tokens per gpu in a minibatch')
parser.add_argument('--beam_size', type=int, default=4, help='Beam size')
parser.add_argument('--lp_alpha', type=float, default=0.6,
                    help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--test_batch_size', type=int, default=256, help='Test batch size')
parser.add_argument('--num_buckets', type=int, default=10, help='Bucket number')
parser.add_argument('--bucket_scheme', type=str, default='constant',
                    help='Strategy for generating bucket keys. It supports: '
                         '"constant": all the buckets have the same width; '
                         '"linear": the width of bucket increases linearly; '
                         '"exp": the width of bucket increases exponentially')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing')
parser.add_argument('--src_max_len', type=int, default=-1, help='Maximum length of the source '
                                                                'sentence, -1 means no clipping')
parser.add_argument('--tgt_max_len', type=int, default=-1, help='Maximum length of the target '
                                                                'sentence, -1 means no clipping')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1.0, help='Initial learning rate')
parser.add_argument('--warmup_steps', type=float, default=4000,
                    help='number of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--num_accumulated', type=int, default=1,
                    help='Number of steps to accumulate the gradients. '
                         'This is useful to mimic large batch training with limited gpu memory')
parser.add_argument('--magnitude', type=float, default=3.0,
                    help='Magnitude of Xavier initialization')
parser.add_argument('--average_checkpoint', action='store_true',
                    help='Turn on to perform final testing based on '
                         'the average of last few checkpoints')
parser.add_argument('--num_averages', type=int, default=5,
                    help='Perform final testing based on the '
                         'average of last num_averages checkpoints. '
                         'This is only used if average_checkpoint is True')
parser.add_argument('--average_start', type=int, default=5,
                    help='Perform average SGD on last average_start epochs')
parser.add_argument('--full', action='store_true',
                    help='In default, we use the test dataset in'
                         ' http://statmt.org/wmt14/test-filtered.tgz.'
                         ' When the option full is turned on, we use the test dataset in'
                         ' http://statmt.org/wmt14/test-full.tgz')
parser.add_argument('--bleu', type=str, default='tweaked',
                    help='Schemes for computing bleu score. It can be: '
                    '"tweaked": it uses similar steps in get_ende_bleu.sh in tensor2tensor '
                    'repository, where compound words are put in ATAT format; '
                    '"13a": This uses official WMT tokenization and produces the same results'
                    ' as official script (mteval-v13a.pl) used by WMT; '
                    '"intl": This use international tokenization in mteval-v14a.pl')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                         '(using single gpu is suggested)')

parser.add_argument('--backbone', type=str, default='DenseNet121',
                    help='backbone cnn architecture')
parser.add_argument('--backbone_from_id', type=str, default=None,
                    help='backbone cnn architecture loading from an ID')
parser.add_argument('--freeze_backbone', type=bool, default=False,
                    help='freeze the backbone cnn')
parser.add_argument('--data_shape', type=int, default=512,
                    help='image shape w,h')
parser.add_argument('--every', type=int, default=5,
                    help='only do every nth frame')
parser.add_argument('--trainable_cnn', type=bool, default=False,
                    help='if false we preextract feats in the transforms stage and dont have a source embedder')

args = parser.parse_args()
logging.info(args)

ctx = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
    [mx.gpu(int(x)) for x in args.gpus.split(',')]
num_ctxs = len(ctx)

backbone_net = get_model(args.backbone, pretrained=True, ctx=ctx).features
cnn_model = FrameModel(backbone_net, 11)  # hardcoded the number of classes
if args.backbone_from_id:
    if os.path.exists(os.path.join('models', 'vision', args.backbone_from_id)):
        files = os.listdir(os.path.join('models', 'vision', args.backbone_from_id))
        files = [f for f in files if f[-7:] == '.params']
        if len(files) > 0:
            files = sorted(files, reverse=True)  # put latest model first
            model_name = files[0]
            cnn_model.load_parameters(os.path.join('models', 'vision', args.backbone_from_id, model_name), ctx=ctx)
            logging.info('Loaded backbone params: {}'.format(os.path.join('models', 'vision', args.backbone_from_id, model_name)))
    else:
        raise FileNotFoundError('{}'.format(os.path.join('models', 'vision', args.backbone_from_id)))

if args.freeze_backbone:
    for param in cnn_model.collect_params().values():
        param.grad_req = 'null'

cnn_model = TimeDistributed(cnn_model.backbone)

if args.trainable_cnn:
    encoder_model = cnn_model
else:
    from mxnet.gluon import nn
    encoder_model = nn.HybridSequential(prefix='src_embed_')
    with encoder_model.name_scope():
        encoder_model.add(nn.Dropout(rate=0.0))

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(args.data_shape),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomLighting(0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(args.data_shape + 32),
    transforms.CenterCrop(args.data_shape),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_train = TennisSet(split='train', transform=transform_train, captions=True, max_cap_len=args.tgt_max_len, every=args.every)
data_val = TennisSet(split='val', transform=transform_test, captions=True, vocab=data_train.vocab, every=args.every)
data_test = TennisSet(split='test', transform=transform_test, captions=True, vocab=data_train.vocab, every=args.every)

val_tgt_sentences = data_val.get_captions(split=True)  # split as bleu set as tweaked otherwise dont split
test_tgt_sentences = data_test.get_captions(split=True)
captioning.write_sentences(val_tgt_sentences, os.path.join('models', 'captioning', args.model_id, 'val_gt.txt'))
captioning.write_sentences(test_tgt_sentences, os.path.join('models', 'captioning', args.model_id, 'test_gt.txt'))

data_train_lengths, data_val_lengths, data_test_lengths = [x.get_data_lens() for x in [data_train, data_val, data_test]]

if args.src_max_len <= 0 or args.tgt_max_len <= 0:
    max_len = np.max([np.max(data_train_lengths, axis=0), np.max(data_val_lengths, axis=0),
                      np.max(data_test_lengths, axis=0)], axis=0)
if args.src_max_len > 0:
    src_max_len = args.src_max_len
else:
    src_max_len = max_len[0]
if args.tgt_max_len > 0:
    tgt_max_len = args.tgt_max_len
else:
    tgt_max_len = max_len[1]

encoder, decoder = get_transformer_encoder_decoder(units=args.num_units,
                                                   hidden_size=args.hidden_size,
                                                   dropout=args.dropout,
                                                   num_layers=args.num_layers,
                                                   num_heads=args.num_heads,
                                                   max_src_length=max(src_max_len, 500),
                                                   max_tgt_length=max(tgt_max_len, 500),
                                                   scaled=args.scaled)
model = NMTModel(src_vocab=None, tgt_vocab=data_train.vocab, encoder=encoder, decoder=decoder,
                 embed_size=args.num_units, #share_embed=True, #tie_weights=True, embed_initializer=None,
                 prefix='transformer_', src_embed=encoder_model)
model.initialize(init=mx.init.Xavier(magnitude=args.magnitude), ctx=ctx)
static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

translator = BeamSearchTranslator(model=model, beam_size=args.beam_size,
                                  scorer=nlp.model.BeamSearchScorer(alpha=args.lp_alpha,
                                                                    K=args.lp_k),
                                  max_length=200)
logging.info('Use beam_size={}, alpha={}, K={}'.format(args.beam_size, args.lp_alpha, args.lp_k))

label_smoothing = LabelSmoothing(epsilon=args.epsilon, units=len(data_train.vocab))
label_smoothing.hybridize(static_alloc=static_alloc)

loss_function = MaskedSoftmaxCELoss(sparse_label=False)
loss_function.hybridize(static_alloc=static_alloc)

test_loss_function = MaskedSoftmaxCELoss()
test_loss_function.hybridize(static_alloc=static_alloc)

rescale_loss = 100
parallel_model = ParallelTransformer(model, label_smoothing, loss_function, rescale_loss)
detokenizer = nlp.data.SacreMosesDetokenizer()


def evaluate(data_loader, context=ctx[0]):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(data_loader):
        src_seq = src_seq.as_in_context(context)
        tgt_seq = tgt_seq.as_in_context(context)
        src_valid_length = src_valid_length.as_in_context(context)
        tgt_valid_length = tgt_valid_length.as_in_context(context)
        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = test_loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)
        # Translate
        samples, _, sample_valid_length = \
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [data_train.vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]
    for ind, sentence in zip(all_inst_ids, translation_out):
        if args.bleu == 'tweaked':
            real_translation_out[ind] = sentence
        elif args.bleu == '13a' or args.bleu == 'intl':
            real_translation_out[ind] = detokenizer(_bpe_to_words(sentence))
        else:
            raise NotImplementedError
    return avg_loss, real_translation_out


def train():
    """Training function."""
    trainer = gluon.Trainer(model.collect_params(), args.optimizer,
                            {'learning_rate': args.lr, 'beta2': 0.98, 'epsilon': 1e-9})

    train_data_loader, val_data_loader, test_data_loader \
        = captioning.get_dataloaders(data_train, data_val, data_test, args,
                                     use_average_length=True, num_shards=len(ctx))

    if args.bleu == 'tweaked':
        bpe = True
        split_compound_word = bpe
        tokenized = True
    elif args.bleu == '13a' or args.bleu == 'intl':
        bpe = False
        split_compound_word = False
        tokenized = False
    else:
        raise NotImplementedError

    best_valid_bleu = 0.0
    step_num = 0
    warmup_steps = args.warmup_steps
    grad_interval = args.num_accumulated
    model.collect_params().setattr('grad_req', 'add')
    average_start = (len(train_data_loader) // grad_interval) * (args.epochs - args.average_start)
    average_param_dict = None
    model.collect_params().zero_grad()
    parallel = Parallel(num_ctxs, parallel_model)
    for epoch_id in range(args.epochs):
        log_avg_loss = 0
        log_wc = 0
        loss_denom = 0
        step_loss = 0
        log_start_time = time.time()
        for batch_id, seqs in enumerate(train_data_loader):
            if batch_id % grad_interval == 0:
                step_num += 1
                new_lr = args.lr / math.sqrt(args.num_units) \
                         * min(1. / math.sqrt(step_num), step_num * warmup_steps ** (-1.5))
                trainer.set_learning_rate(new_lr)
            src_wc, tgt_wc, bs = np.sum([(shard[2].sum(), shard[3].sum(), shard[0].shape[0]) for shard in seqs], axis=0)
            seqs = [[seq.as_in_context(context) for seq in shard]
                    for context, shard in zip(ctx, seqs)]
            Ls = []
            for seq in seqs:
                if not args.trainable_cnn:  # for memory reasons we run through the encoder here and not in the RNN, can't back prop
                    seq[0] = cnn_model(seq[0])
                parallel.put((seq, args.batch_size))
            Ls = [parallel.get() for _ in range(len(ctx))]
            src_wc = src_wc.asscalar()
            tgt_wc = tgt_wc.asscalar()
            loss_denom += tgt_wc - bs
            if batch_id % grad_interval == grad_interval - 1 or\
                    batch_id == len(train_data_loader) - 1:
                if average_param_dict is None:
                    average_param_dict = {k: v.data(ctx[0]).copy() for k, v in
                                          model.collect_params().items()}
                trainer.step(float(loss_denom) / args.batch_size / 100.0)
                param_dict = model.collect_params()
                param_dict.zero_grad()
                if step_num > average_start:
                    alpha = 1. / max(1, step_num - average_start)
                    for name, average_param in average_param_dict.items():
                        average_param[:] += alpha * (param_dict[name].data(ctx[0]) - average_param)
            step_loss += sum([L.asscalar() for L in Ls])
            if batch_id % grad_interval == grad_interval - 1 or\
                    batch_id == len(train_data_loader) - 1:
                log_avg_loss += step_loss / loss_denom * args.batch_size * 100.0
                loss_denom = 0
                step_loss = 0
            log_wc += src_wc + tgt_wc
            if (batch_id + 1) % (args.log_interval * grad_interval) == 0:
                wps = log_wc / (time.time() - log_start_time)
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'
                             .format(epoch_id, batch_id + 1, len(train_data_loader),
                                     log_avg_loss / args.log_interval,
                                     np.exp(log_avg_loss / args.log_interval),
                                     wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0
        mx.nd.waitall()
        valid_loss, valid_translation_out = evaluate(val_data_loader, ctx[0])
        valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out,
                                                    tokenized=tokenized, tokenizer=args.bleu,
                                                    split_compound_word=split_compound_word,
                                                    bpe=bpe)
        logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                     .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
        test_loss, test_translation_out = evaluate(test_data_loader, ctx[0])
        test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out,
                                                   tokenized=tokenized, tokenizer=args.bleu,
                                                   split_compound_word=split_compound_word,
                                                   bpe=bpe)
        logging.info('[Epoch {}] test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                     .format(epoch_id, test_loss, np.exp(test_loss), test_bleu_score * 100))
        captioning.write_sentences(valid_translation_out,
                                   os.path.join('models', 'captioning', args.model_id,
                                                   'epoch{:d}_valid_out.txt').format(epoch_id))
        captioning.write_sentences(test_translation_out,
                                   os.path.join('models', 'captioning', args.model_id,
                                                   'epoch{:d}_test_out.txt').format(epoch_id))
        if valid_bleu_score > best_valid_bleu:
            best_valid_bleu = valid_bleu_score
            save_path = os.path.join('models', 'captioning', args.model_id, 'valid_best.params')
            logging.info('Save best parameters to {}'.format(save_path))
            model.save_parameters(save_path)
        save_path = os.path.join('models', 'captioning', args.model_id, 'epoch{:d}.params'.format(epoch_id))
        model.save_parameters(save_path)
    save_path = os.path.join('models', 'captioning', args.model_id, 'average.params')
    mx.nd.save(save_path, average_param_dict)
    if args.average_checkpoint:
        for j in range(args.num_averages):
            params = mx.nd.load(os.path.join('models', 'captioning', args.model_id,
                                             'epoch{:d}.params'.format(args.epochs - j - 1)))
            alpha = 1. / (j + 1)
            for k, v in model._collect_params_with_prefix().items():
                for c in ctx:
                    v.data(c)[:] += alpha * (params[k].as_in_context(c) - v.data(c))
        save_path = os.path.join('models', 'captioning', args.model_id,
                                 'average_checkpoint_{}.params'.format(args.num_averages))
        model.save_parameters(save_path)
    elif args.average_start > 0:
        for k, v in model.collect_params().items():
            v.set_data(average_param_dict[k])
        save_path = os.path.join('models', 'captioning', args.model_id, 'average.params')
        model.save_parameters(save_path)
    else:
        model.load_parameters(os.path.join('models', 'captioning', args.model_id, 'valid_best.params'), ctx)
    valid_loss, valid_translation_out = evaluate(val_data_loader, ctx[0])
    valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out,
                                                tokenized=tokenized, tokenizer=args.bleu, bpe=bpe,
                                                split_compound_word=split_compound_word)
    logging.info('Best model valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                 .format(valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
    test_loss, test_translation_out = evaluate(test_data_loader, ctx[0])
    test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out,
                                               tokenized=tokenized, tokenizer=args.bleu, bpe=bpe,
                                               split_compound_word=split_compound_word)
    logging.info('Best model test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                 .format(test_loss, np.exp(test_loss), test_bleu_score * 100))
    captioning.write_sentences(valid_translation_out,
                               os.path.join('models', 'captioning', args.model_id, 'best_valid_out.txt'))
    captioning.write_sentences(test_translation_out,
                               os.path.join('models', 'captioning', args.model_id, 'best_test_out.txt'))


if __name__ == '__main__':
    train()
