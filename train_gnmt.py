"""
Google Neural Machine Translation
=================================

@article{wu2016google,
  title={Google's neural machine translation system:
   Bridging the gap between human and machine translation},
  author={Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc V and
   Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and
   Macherey, Klaus and others},
  journal={arXiv preprint arXiv:1609.08144},
  year={2016}
}
"""


from absl import app, flags
from absl.flags import FLAGS
import time
import random
import os
import io
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

from gluonnlp.model.translation import NMTModel
from gluonnlp.loss import MaskedSoftmaxCELoss
from models.captioning.gnmt import get_gnmt_encoder_decoder
from utils.translation import BeamSearchTranslator
from metrics.bleu import compute_bleu
from utils import captioning
from dataset import TennisSet
from models.vision.definitions import FrameModel
from utils.layers import TimeDistributed
from gluoncv.model_zoo import get_model
from mxnet.gluon.data.vision import transforms

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

flags.DEFINE_string('model_id', '0000',
                    'model identification string')
flags.DEFINE_integer('epochs', 40,
                     'How many training epochs to complete')
flags.DEFINE_integer('num_hidden', 128,
                     'Dimension of the embedding vectors and states')
flags.DEFINE_float('dropout', 0.2,
                   'dropout applied to layers (0 = no dropout)')
flags.DEFINE_integer('num_layers', 2,
                     'Number of layers in the encoder  and decoder')
flags.DEFINE_integer('num_bi_layers', 1,
                     'Number of bidirectional layers in the encoder and decoder')

flags.DEFINE_integer('batch_size', 128,
                     'Batch size for detection: higher faster, but more memory intensive.')

flags.DEFINE_integer('beam_size', 4,
                     'Beam size.')

flags.DEFINE_float('lp_alpha', 1.0,
                   'Alpha used in calculating the length penalty')
flags.DEFINE_integer('lp_k', 5,
                     'K used in calculating the length penalty')
flags.DEFINE_integer('test_batch_size', 32,
                     'Test batch size')
flags.DEFINE_integer('num_buckets', 5,
                     'Bucket number')

flags.DEFINE_string('bucket_scheme', 'constant',
                    'Strategy for generating bucket keys. It supports: '
                    '"constant": all the buckets have the same width; '
                    '"linear": the width of bucket increases linearly; '
                    '"exp": the width of bucket increases exponentially')


flags.DEFINE_float('bucket_ratio', 0.0,
                   'Ratio for increasing the throughput of the bucketing')
flags.DEFINE_integer('tgt_max_len', 50,
                     'Maximum length of the target sentence')
flags.DEFINE_string('optimizer', 'adam',
                    'optimization algorithm')
flags.DEFINE_float('lr', 1E-3,
                   'Initial learning rate')
flags.DEFINE_float('lr_update_factor', 0.5,
                   'Learning rate decay factor')
flags.DEFINE_float('clip', 5.0,
                   'gradient clipping')

flags.DEFINE_integer('log_interval', 100,
                     'Logging mini-batch interval.')

flags.DEFINE_integer('num_gpus', 1,
                     'Number of GPUs to use')

flags.DEFINE_string('backbone', 'DenseNet121',
                    'Backbone CNN name')
flags.DEFINE_string('backbone_from_id',  None,
                    'Load a backbone model from a model_id, used for Temporal Pooling with fine-tuned CNN')
flags.DEFINE_bool('freeze_backbone', False,
                  'Freeze the backbone model')
flags.DEFINE_integer('data_shape', 512,
                     'The width and height for the input image to be cropped to.')
flags.DEFINE_integer('every', 5,
                     'Use only every this many frames: [train, val, test] splits')
flags.DEFINE_string('feats_model', None,
                    'load CNN features as npy files from this model')


def main(_argv):

    # ctx = [mx.gpu(i) for i in range(FLAGS.num_gpus)] if FLAGS.num_gpus > 0 else [mx.cpu()]
    if FLAGS.num_gpus > 0:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()

    if FLAGS.feats_model is None:
        backbone_net = get_model(FLAGS.backbone, pretrained=True, ctx=ctx).features
        cnn_model = FrameModel(backbone_net, 11)  # hardcoded the number of classes
        if FLAGS.backbone_from_id:
            if os.path.exists(os.path.join('models', 'vision', FLAGS.backbone_from_id)):
                files = os.listdir(os.path.join('models', 'vision', FLAGS.backbone_from_id))
                files = [f for f in files if f[-7:] == '.params']
                if len(files) > 0:
                    files = sorted(files, reverse=True)  # put latest model first
                    model_name = files[0]
                    cnn_model.load_parameters(os.path.join('models', 'vision', FLAGS.backbone_from_id, model_name), ctx=ctx)
                    logging.info('Loaded backbone params: {}'.format(os.path.join('models', 'vision', FLAGS.backbone_from_id, model_name)))
            else:
                raise FileNotFoundError('{}'.format(os.path.join('models', 'vision', FLAGS.backbone_from_id)))

        if FLAGS.freeze_backbone:
            for param in cnn_model.collect_params().values():
                param.grad_req = 'null'

        cnn_model = TimeDistributed(cnn_model.backbone)

        encoder_model = cnn_model

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(FLAGS.data_shape),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomLighting(0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(FLAGS.data_shape + 32),
            transforms.CenterCrop(FLAGS.data_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    else:
        from mxnet.gluon import nn
        encoder_model = nn.HybridSequential(prefix='src_embed_')
        with encoder_model.name_scope():
            encoder_model.add(nn.Dropout(rate=0.0))

        transform_train = None
        transform_test = None

    data_train = TennisSet(split='train', transform=transform_train, captions=True, max_cap_len=FLAGS.tgt_max_len,
                           every=FLAGS.every, feats_model=FLAGS.feats_model)
    data_val = TennisSet(split='val', transform=transform_test, captions=True, vocab=data_train.vocab,
                         every=FLAGS.every, inference=True, feats_model=FLAGS.feats_model)
    data_test = TennisSet(split='test', transform=transform_test, captions=True, vocab=data_train.vocab,
                          every=FLAGS.every, inference=True, feats_model=FLAGS.feats_model)

    val_tgt_sentences = data_val.get_captions(split=True)
    test_tgt_sentences = data_test.get_captions(split=True)
    captioning.write_sentences(val_tgt_sentences, os.path.join('models', 'captioning', FLAGS.model_id, 'val_gt.txt'))
    captioning.write_sentences(test_tgt_sentences, os.path.join('models', 'captioning', FLAGS.model_id, 'test_gt.txt'))

    encoder, decoder = get_gnmt_encoder_decoder(hidden_size=FLAGS.num_hidden,
                                                dropout=FLAGS.dropout,
                                                num_layers=FLAGS.num_layers,
                                                num_bi_layers=FLAGS.num_bi_layers)
    model = NMTModel(src_vocab=None, tgt_vocab=data_train.vocab, encoder=encoder, decoder=decoder,
                     embed_size=FLAGS.num_hidden, prefix='gnmt_', src_embed=encoder_model)
    model.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
    static_alloc = True
    model.hybridize(static_alloc=static_alloc)
    logging.info(model)

    translator = BeamSearchTranslator(model=model, beam_size=FLAGS.beam_size,
                                      scorer=nlp.model.BeamSearchScorer(alpha=FLAGS.lp_alpha,
                                                                        K=FLAGS.lp_k),
                                      max_length=FLAGS.tgt_max_len + 100)
    logging.info('Use beam_size={}, alpha={}, K={}'.format(FLAGS.beam_size, FLAGS.lp_alpha, FLAGS.lp_k))

    loss_function = MaskedSoftmaxCELoss()
    loss_function.hybridize(static_alloc=static_alloc)

    train(data_train, data_val, data_test, model, loss_function, val_tgt_sentences, test_tgt_sentences, translator, ctx)


def evaluate(data_loader, model, loss_function, translator, data_train, ctx):
    """Evaluate
    """
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    for batch_id, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) in enumerate(data_loader):
        # if batch_id == len(data_loader)-1:
        #     break  # errors on last batch, jump out for now
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)

        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)

        # Translate
        samples, _, sample_valid_length =\
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
        real_translation_out[ind] = sentence

    return avg_loss, real_translation_out


def train(data_train, data_val, data_test, model, loss_function, val_tgt_sentences, test_tgt_sentences, translator, ctx):
    """Training function.
    """

    trainer = gluon.Trainer(model.collect_params(), FLAGS.optimizer, {'learning_rate': FLAGS.lr})

    train_data_loader, val_data_loader, test_data_loader = captioning.get_dataloaders(data_train, data_val, data_test)

    best_valid_bleu = 0.0
    for epoch_id in range(FLAGS.epochs):
        log_avg_loss = 0
        log_wc = 0
        log_start_time = time.time()
        for batch_id, (src_seq, tgt_seq, src_valid_length, tgt_valid_length) in enumerate(train_data_loader):
            if batch_id == len(train_data_loader)-1:
                break  # errors on last batch, jump out for now
            # logging.info(src_seq.context) Context suddenly becomes GPU.
            src_seq = src_seq.as_in_context(ctx)
            tgt_seq = tgt_seq.as_in_context(ctx)
            src_valid_length = src_valid_length.as_in_context(ctx)
            tgt_valid_length = tgt_valid_length.as_in_context(ctx)

            with mx.autograd.record():
                out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
                loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean()
                loss = loss * (tgt_seq.shape[1] - 1) / (tgt_valid_length - 1).mean()
                loss.backward()

            trainer.step(1)
            src_wc = src_valid_length.sum().asscalar()
            tgt_wc = (tgt_valid_length - 1).sum().asscalar()
            step_loss = loss.asscalar()
            log_avg_loss += step_loss
            log_wc += src_wc + tgt_wc
            if (batch_id + 1) % FLAGS.log_interval == 0:
                wps = log_wc / (time.time() - log_start_time)
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}  throughput={:.2f}K wps, wc={:.2f}K'
                             .format(epoch_id, batch_id + 1, len(train_data_loader),
                                     log_avg_loss / FLAGS.log_interval,
                                     np.exp(log_avg_loss / FLAGS.log_interval),
                                     wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0

        valid_loss, valid_translation_out = evaluate(val_data_loader, model, loss_function, translator, data_train, ctx)
        valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out)
        logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'.format(epoch_id,
                                                                                                valid_loss,
                                                                                                np.exp(valid_loss),
                                                                                                valid_bleu_score * 100))

        test_loss, test_translation_out = evaluate(test_data_loader, model, loss_function, translator, data_train, ctx)
        test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out)
        logging.info('[Epoch {}] test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'.format(epoch_id,
                                                                                             test_loss,
                                                                                             np.exp(test_loss),
                                                                                             test_bleu_score * 100))

        captioning.write_sentences(valid_translation_out, os.path.join('models', 'captioning', FLAGS.model_id,
                                                                       'epoch{:d}_valid_out.txt').format(epoch_id))
        captioning.write_sentences(test_translation_out, os.path.join('models', 'captioning', FLAGS.model_id,
                                                                      'epoch{:d}_test_out.txt').format(epoch_id))
        if valid_bleu_score > best_valid_bleu:
            best_valid_bleu = valid_bleu_score
            save_path = os.path.join('models', 'captioning', FLAGS.model_id, 'valid_best.params')
            logging.info('Save best parameters to {}'.format(save_path))
            model.save_parameters(save_path)

        if epoch_id + 1 >= (FLAGS.epochs * 2) // 3:
            new_lr = trainer.learning_rate * FLAGS.lr_update_factor
            logging.info('Learning rate change to {}'.format(new_lr))
            trainer.set_learning_rate(new_lr)

    if os.path.exists(os.path.join('models', 'captioning', FLAGS.model_id, 'valid_best.params')):
        model.load_parameters(os.path.join('models', 'captioning', FLAGS.model_id, 'valid_best.params'))

    valid_loss, valid_translation_out = evaluate(val_data_loader, model, loss_function, translator, data_train, ctx)
    valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out)
    logging.info('Best model valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'.format(valid_loss,
                                                                                            np.exp(valid_loss),
                                                                                            valid_bleu_score * 100))

    test_loss, test_translation_out = evaluate(test_data_loader, model, loss_function, translator, data_train, ctx)
    test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out)
    logging.info('Best model test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}' .format(test_loss,
                                                                                          np.exp(test_loss),
                                                                                          test_bleu_score * 100))

    captioning.write_sentences(valid_translation_out,
                               os.path.join('models', 'captioning', FLAGS.model_id, 'best_valid_out.txt'))
    captioning.write_sentences(test_translation_out,
                               os.path.join('models', 'captioning', FLAGS.model_id, 'best_test_out.txt'))


def write_sentences(sentences, file_path):
    with io.open(file_path, 'w', encoding='utf-8') as of:
        for sent in sentences:
            if isinstance(sent, (list, tuple)):
                of.write(u' '.join(sent) + u'\n')
            else:
                of.write(sent + u'\n')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
