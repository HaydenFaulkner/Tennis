"""Train script"""
from absl import app, flags
from absl.flags import FLAGS
import logging
import multiprocessing
import mxnet as mx
import numpy as np
import os
import shutil
import sys
from tensorboardX import SummaryWriter
import time

from mxnet import gluon, init
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model

from model import TimeModel, FrameModel
from dataset import TennisSet
# from utils import frames_to_video

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

flags.DEFINE_string('backbone', 'resnet18_v1',
                    'Backbone CNN name: resnet18_v1')
flags.DEFINE_string('model_id', '0000',
                    'model identification string')
flags.DEFINE_string('split_id', '01',
                    'split identification string, 01: single test vid; 02: all videos have test sections')
flags.DEFINE_integer('log_interval', 100,
                     'Logging mini-batch interval.')

flags.DEFINE_list('window', '1, 1',
                  'Temporal window size of frames and the frame gap of the windows samples')
flags.DEFINE_list('every', '1, 1, 1',
                  'Use only every this many frames: [train, val, test] splits')
flags.DEFINE_list('padding', '1, 1, 1',
                  'Frame*every + and - padding around the marked event boundaries: [train, val, test] splits')
flags.DEFINE_list('balance', 'True, False, False',
                  'Balance the play/not class samples: [train, val, test] splits')

flags.DEFINE_integer('batch_size', 128,
                     'Batch size for detection: higher faster, but more memory intensive.')
flags.DEFINE_integer('epochs', 10,
                     'How many training epochs to complete')
flags.DEFINE_integer('num_gpus', 1,
                     'Number of GPUs to use')
flags.DEFINE_integer('num_workers', -1,
                     'The number of workers should be picked so that it’s equal to number of cores on your machine '
                     'for max parallelization. If this number is bigger than your number of cores it will use up '
                     'a bunch of extra CPU memory. -1 is auto.')

flags.DEFINE_float('lr', 0.001,
                   'Learning rate.')
flags.DEFINE_float('lr_factor', 0.75,
                   'lr factor.')
flags.DEFINE_list('lr_steps', '10',
                  'Epochs at which learning rate factor applied.')
flags.DEFINE_float('momentum', 0.9,
                   'momentum.')
flags.DEFINE_float('wd', 0.0001,
                   'weight decay.')


def main(_argv):
    FLAGS.window = [int(s) for s in FLAGS.window]
    FLAGS.every = [int(s) for s in FLAGS.every]
    FLAGS.padding = [int(s) for s in FLAGS.padding]
    FLAGS.balance = [True if s.lower() == 'true' or s.lower() == 't' else False for s in FLAGS.balance]
    FLAGS.lr_steps = [int(s) for s in FLAGS.lr_steps]

    if FLAGS.num_workers < 0:
        FLAGS.num_workers = multiprocessing.cpu_count()

    ctx = [mx.gpu(i) for i in range(FLAGS.num_gpus)] if FLAGS.num_gpus > 0 else [mx.cpu()]
    classes = 2

    # Set up logging
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join('models', FLAGS.model_id, 'log.txt')
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
    logging.info('\n'.join(f.serialize() for f in key_flags))

    # set up tensorboard summary writer
    tb_sw = SummaryWriter(log_dir=os.path.join(log_dir, 'tb'), comment=FLAGS.model_id)

    # Data augmentation, will do in dataset incase window>1 and need to be applied image-wise
    jitter_param = 0.4
    lighting_param = 0.1

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_set = TennisSet(split='train', transform=transform_train, every=FLAGS.every[0], padding=FLAGS.padding[0],
                          window=FLAGS.window, model_id=FLAGS.model_id, balance=True, split_id=FLAGS.split_id)
    val_set = TennisSet(split='val', transform=transform_test, every=FLAGS.every[1], padding=FLAGS.padding[1],
                        window=FLAGS.window, model_id=FLAGS.model_id, balance=False, split_id=FLAGS.split_id)
    test_set = TennisSet(split='test', transform=transform_test, every=FLAGS.every[2], padding=FLAGS.padding[2],
                         window=FLAGS.window, model_id=FLAGS.model_id, balance=False, split_id=FLAGS.split_id)

    logging.info(train_set)
    logging.info(val_set)
    logging.info(test_set)

    # Data Loaders
    train_data = gluon.data.DataLoader(train_set, batch_size=FLAGS.batch_size,
                                       shuffle=True, num_workers=FLAGS.num_workers)
    val_data = gluon.data.DataLoader(val_set, batch_size=FLAGS.batch_size,
                                     shuffle=False, num_workers=FLAGS.num_workers)
    test_data = gluon.data.DataLoader(test_set, batch_size=FLAGS.batch_size,
                                      shuffle=False, num_workers=FLAGS.num_workers)

    # Define Model
    finetune_net = get_model(FLAGS.backbone, pretrained=True)

    if FLAGS.window[0] == 1:
        # # original
        # model = finetune_net
        # with model.name_scope():
        #     model.output = nn.Dense(classes)
        # model.output.initialize(init.Xavier(), ctx=ctx)
        # model.collect_params().reset_ctx(ctx)
        # model.hybridize()
        # original
        model = FrameModel(finetune_net, classes)
        model.initialize(init.Xavier(), ctx=ctx)
        model.collect_params().reset_ctx(ctx)
        model.hybridize()
    else:
        # Time Distributed RNN
        with finetune_net.name_scope():
            finetune_net.output = nn.Dense(256)
        finetune_net.output.initialize(init.Xavier(), ctx=ctx)

        model = TimeModel(finetune_net, classes)
        model.initialize(init.Xavier(), ctx)
        model.collect_params().reset_ctx(ctx)
        # model.hybridize()  # hybridize doesn't work for this model

    start_epoch = 0
    if os.path.exists(os.path.join('models', FLAGS.model_id)):
        files = os.listdir(os.path.join('models', FLAGS.model_id))
        files = [f for f in files if f[-7:] == '.params']
        if len(files) > 0:
            files = sorted(files, reverse=True)  # put latest model first
            model_name = files[0]
            start_epoch = int(model_name.split('_')[0]) + 1
            model.load_parameters(os.path.join('models', FLAGS.model_id, model_name), ctx=ctx)
            logging.info('Loaded model params: {}'.format(os.path.join('models', FLAGS.model_id, model_name)))

    # Setup the optimiser
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': FLAGS.lr, 'momentum': FLAGS.momentum, 'wd': FLAGS.wd})

    # Setup Metric/s
    metrics = [(('acc', mx.metric.Accuracy()), )]
    val_metrics = [(('acc', mx.metric.Accuracy()), )]
    test_metrics = [(('acc', mx.metric.Accuracy()), )]

    # Setup Loss/es
    play_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # Testing/Validation function
    def test_model(net, loader, dataset, metrics, ctx, vis=False):

        for i, batch in enumerate(loader):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            idxs = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)
            outputs = [net(x) for x in data]

            if len(metrics) > 1:
                for li, metric_set in enumerate(metrics):
                    for mi, metric in enumerate(metric_set):
                        metric[1].update([l[:, li] for l in labels], [o[li] for o in outputs])
            else:
                for mi, metric in enumerate(metrics[0]):
                    metric[1].update(labels, outputs)

            if vis:
                # save the images with labels
                for di in range(len(outputs)):  # loop over devices
                    idxs = [int(idx) for idx in idxs[di].asnumpy()]

                    output = [o.asnumpy() for o in outputs[di]]
                    if isinstance(outputs[0], list) or isinstance(outputs[0], tuple):
                        for i in range(len(idxs)):  # loop over samples
                            dataset.save_sample(idxs[i], [o[i] for o in output])
                    else:
                        for i in range(len(idxs)):  # loop over samples
                            dataset.save_sample(idxs[i], output[i])

        results = []
        for metric_set in metrics:  # loop through the number or losses/metrics
            metric_results = []
            for metric in metric_set:  # loop through acc and F1
                metric_results.append(metric[1].get()[1])
            results.append(metric_results)

        return results

    # Training loop
    lr_counter = 0
    num_batches = int(len(train_set)/FLAGS.batch_size)
    for epoch in range(start_epoch, FLAGS.epochs):  # loop over epochs
        logging.info('[Starting Epoch {}]'.format(epoch))
        if epoch == FLAGS.lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*FLAGS.lr_factor)
            lr_counter += 1

        tic = time.time()
        train_sum_loss = 0
        for metric_set in metrics:
            for metric in metric_set:
                metric[1].reset()

        for i, batch in enumerate(train_data):  # loop over batches
            btic = time.time()

            # split data across devices
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            sum_losses = []
            outputs = []
            with ag.record():
                for ix, x in enumerate(data):  # loop over devices
                    output = model(x)
                    outputs.append(output)
                    sum_losses.append(play_loss(output, labels[ix]))

                ag.backward(sum_losses)

            # step the optimizer
            trainer.step(FLAGS.batch_size)

            # store the epoch loss sums - avg loss across batch (avg across devices)
            train_sum_loss += sum([l.mean().asscalar() for l in sum_losses]) / len(sum_losses)

            # update metric
            if len(metrics) > 1:
                for li, metric_set in enumerate(metrics):
                    for mi, metric in enumerate(metric_set):
                        metric[1].update([l[:, li] for l in labels], [o[li] for o in outputs])
            else:
                for mi, metric in enumerate(metrics[0]):
                    metric[1].update(labels, outputs)

            # logging
            if FLAGS.log_interval and not (i + 1) % FLAGS.log_interval:
                str_ = '[Epoch {}][Batch {}/{}], LR: {:.2E}, Speed: {:.3f} samples/sec'.format(
                    epoch, i, num_batches, trainer.learning_rate, FLAGS.batch_size / (time.time() - btic))

                str_ += ', {}={:.3f}'.format("loss:", train_sum_loss/(i*FLAGS.batch_size))
                tb_sw.add_scalar(tag='Training_loss',
                                 scalar_value=train_sum_loss/(i*FLAGS.batch_size),
                                 global_step=(epoch * len(train_data) + i))
                for mi, metric in enumerate(metrics[0]):
                    str_ += ', {}={:.3f}'.format(metric[0], metric[1].get()[1])
                    tb_sw.add_scalar(tag='Training_{}'.format(metric[0]),
                                     scalar_value=float(metric[1].get()[1]),
                                     global_step=(epoch * len(train_data) + i))
                logging.info(str_)

        # Format end of epoch logging string getting metrics along the way
        str_ = '[Epoch {}]'.format(epoch)

        for li, metric_set in enumerate(metrics):
            for mi, metric in enumerate(metric_set):
                str_ += ', Train_{}={:.3f}'.format(metric[0], metric[1].get()[1])

        str_ += ', loss: {:.3f}'.format(train_sum_loss / len(train_data))

        vtic = time.time()
        val_accs = test_model(model, val_data, val_set, val_metrics, ctx)
        for li, metric_set in enumerate(val_metrics):
            for mi, metric in enumerate(metric_set):
                str_ += ', Val_{}={:.3f}'.format(metric[0], val_accs[li][mi])
                tb_sw.add_scalar(tag='Val_{}'.format(metric[0]),
                                 scalar_value=float(val_accs[li][mi]),
                                 global_step=(epoch * len(train_data)))

        str_ += ', Epoch Time: {:.1f}, Val Time: {:.1f}'.format(time.time() - tic, time.time() - vtic)

        logging.info(str_)

        model.save_parameters(os.path.join('models', FLAGS.model_id,
                                           "{:04d}_{:.4f}.params".format(epoch, val_accs[0][0])))

    # model training complete, test it
    tic = time.time()
    test_accs = test_model(model, test_data, test_set, test_metrics, ctx, vis=False)

    str_ = '[Finished] '
    for li, metric_set in enumerate(test_metrics):
        for mi, metric in enumerate(metric_set):
            str_ += ', Test_{}={:.3f}'.format(metric[0], test_accs[li][mi])

    str_ += '  # Samples: {}, Time Taken: {:.1f}'.format(len(test_set), time.time() - tic)
    logging.info(str_)

    # logging.info("Cleaning up, making test videos.")
    # for video in os.listdir(test_set.output_dir):
    #     frames_to_video(os.path.join(test_set.output_dir, video), os.path.join(test_set.output_dir, video[:-4]),
    #                     fps=int(25/FLAGS.every[2]))
    #     shutil.rmtree(os.path.join(test_set.output_dir, video))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
