"""Train script"""
from absl import app, flags
from absl.flags import FLAGS
import logging
import multiprocessing
import mxnet as mx
import numpy as np
import os
import sys
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import warnings

from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.accuracy import Accuracy

from models.vision.definitions import CNNRNN, FrameModel, TwoStreamModel, TemporalPooling
from dataset import TennisSet
from metrics.vision import PRF1
from models.vision.rdnet.r21d import get_r21d
# from utils import frames_to_video

from utils.transforms import TwoStreamNormalize

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

flags.DEFINE_string('backbone', 'resnet18_v2',
                    'Backbone CNN name: resnet18_v1')
flags.DEFINE_string('backbone_from_id',  None,
                    'Load a backbone model from a model_id, used for Temporal Pooling with fine-tuned CNN')
flags.DEFINE_bool('freeze_backbone', False,
                  'Freeze the backbone model')
flags.DEFINE_string('model_id', '0000',
                    'model identification string')
flags.DEFINE_string('split_id', '02',
                    'split identification string, 01: single test vid; 02: all videos have test sections')
flags.DEFINE_integer('log_interval', 100,
                     'Logging mini-batch interval.')
flags.DEFINE_integer('data_shape', 512, #224,
                     'The width and height for the input image to be cropped to.')

flags.DEFINE_list('every', '1, 1, 1',
                  'Use only every this many frames: [train, val, test] splits')
flags.DEFINE_list('balance', 'True, False, False',
                  'Balance the play/not class samples: [train, val, test] splits')
flags.DEFINE_integer('window', 1,
                     'Temporal window size of frames')
flags.DEFINE_integer('padding', 1,
                     'Frame*every + and - padding around the marked event boundaries: [train, val, test] splits')
flags.DEFINE_integer('stride', 1,
                     'Temporal stride of samples within a window')

flags.DEFINE_integer('batch_size', 64,
                     'Batch size for detection: higher faster, but more memory intensive.')
flags.DEFINE_integer('epochs', 20,
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
flags.DEFINE_list('lr_steps', '10, 20',
                  'Epochs at which learning rate factor applied.')
flags.DEFINE_float('momentum', 0.9,
                   'momentum.')
flags.DEFINE_float('wd', 0.0001,
                   'weight decay.')

flags.DEFINE_bool('vis', False,
                  'Visualise testing results')
flags.DEFINE_bool('save_feats', False,
                  'save CNN features as npy files')
flags.DEFINE_string('feats_model', None,
                    'load CNN features as npy files from this model')

flags.DEFINE_string('flow', '',
                    'How to use flow, "" for none, "only" for no rgb, "sixc" for six channel inp, "twos" for twostream')
flags.DEFINE_string('temp_pool', None,
                    'mean, max or gru.')

flags.DEFINE_integer('max_batches', -1,  # for 0031
                     'Only do this many batches then break')


def main(_argv):
    FLAGS.every = [int(s) for s in FLAGS.every]
    FLAGS.balance = [True if s.lower() == 'true' or s.lower() == 't' else False for s in FLAGS.balance]
    FLAGS.lr_steps = [int(s) for s in FLAGS.lr_steps]

    if FLAGS.num_workers < 0:
        FLAGS.num_workers = multiprocessing.cpu_count()

    ctx = [mx.gpu(i) for i in range(FLAGS.num_gpus)] if FLAGS.num_gpus > 0 else [mx.cpu()]

    # Set up logging
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join('models', 'vision', 'experiments', FLAGS.model_id, 'log.txt')
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
    logging.info('\n'.join(f.serialize() for f in key_flags))

    # set up tensorboard summary writer
    tb_sw = SummaryWriter(log_dir=os.path.join(log_dir, 'tb'), comment=FLAGS.model_id)

    feat_sub_dir = None

    # Data augmentation, will do in dataset incase window>1 and need to be applied image-wise
    jitter_param = 0.4
    lighting_param = 0.1
    transform_train = None
    transform_test = None
    balance_train = True
    if FLAGS.feats_model is None:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(FLAGS.data_shape),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                         saturation=jitter_param),
            transforms.RandomLighting(lighting_param),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(FLAGS.data_shape + 32),
            transforms.CenterCrop(FLAGS.data_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if bool(FLAGS.flow):

            transform_test = transforms.Compose([
                transforms.Resize(FLAGS.data_shape + 32),
                transforms.CenterCrop(FLAGS.data_shape),
                TwoStreamNormalize()
            ])

            transform_train = transform_test

    if FLAGS.save_feats:
        balance_train = False
        transform_train = transform_test

    if FLAGS.window > 1:
        transform_train = transform_test

    # Load datasets
    if FLAGS.temp_pool not in ['max', 'mean']:
        train_set = TennisSet(split='train', transform=transform_train, every=FLAGS.every[0], padding=FLAGS.padding,
                              stride=FLAGS.stride, window=FLAGS.window, model_id=FLAGS.model_id, split_id=FLAGS.split_id,
                              balance=balance_train, flow=bool(FLAGS.flow), feats_model=FLAGS.feats_model, save_feats=FLAGS.save_feats)

        logging.info(train_set)

        val_set = TennisSet(split='val', transform=transform_test, every=FLAGS.every[1], padding=FLAGS.padding,
                            stride=FLAGS.stride, window=FLAGS.window, model_id=FLAGS.model_id, split_id=FLAGS.split_id,
                            balance=False, flow=bool(FLAGS.flow), feats_model=FLAGS.feats_model, save_feats=FLAGS.save_feats)

        logging.info(val_set)

    test_set = TennisSet(split='test', transform=transform_test, every=FLAGS.every[2], padding=FLAGS.padding,
                         stride=FLAGS.stride, window=FLAGS.window, model_id=FLAGS.model_id, split_id=FLAGS.split_id,
                         balance=False, flow=bool(FLAGS.flow), feats_model=FLAGS.feats_model, save_feats=FLAGS.save_feats)

    logging.info(test_set)

    # Data Loaders
    if FLAGS.temp_pool not in ['max', 'mean']:
        train_data = gluon.data.DataLoader(train_set, batch_size=FLAGS.batch_size,
                                           shuffle=True, num_workers=FLAGS.num_workers)
        val_data = gluon.data.DataLoader(val_set, batch_size=FLAGS.batch_size,
                                         shuffle=False, num_workers=FLAGS.num_workers)
    test_data = gluon.data.DataLoader(test_set, batch_size=FLAGS.batch_size,
                                      shuffle=False, num_workers=FLAGS.num_workers)

    # Define Model
    model = None
    if FLAGS.feats_model is None:
        if FLAGS.backbone == 'rdnet':
            backbone_net = get_r21d(num_layers=34, n_classes=400, t=8, pretrained=True).features
        else:
            if FLAGS.flow == 'sixc':
                backbone_net = get_model(FLAGS.backbone, pretrained=False).features  # 6 channel input, don't want pretraind
            else:
                backbone_net = get_model(FLAGS.backbone, pretrained=True).features

        if FLAGS.flow in ['twos', 'only']:
            if FLAGS.flow == 'only':
                backbone_net = None
            flow_net = get_model(FLAGS.backbone, pretrained=True).features  # todo orig exp was not pretrained flow
            model = TwoStreamModel(backbone_net, flow_net, len(train_set.classes))
        elif FLAGS.backbone == 'rdnet':
            model = FrameModel(backbone_net, len(train_set.classes), swap=True)
        else:
            model = FrameModel(backbone_net, len(train_set.classes))
    elif FLAGS.temp_pool in ['max', 'mean']:
        backbone_net = get_model(FLAGS.backbone, pretrained=True).features
        model = FrameModel(backbone_net, len(test_set.classes))
    if FLAGS.window > 1:  # Time Distributed RNN

        if FLAGS.backbone_from_id and model is not None:
            if os.path.exists(os.path.join('models', 'vision', 'experiments', FLAGS.backbone_from_id)):
                files = os.listdir(os.path.join('models', 'vision', 'experiments', FLAGS.backbone_from_id))
                files = [f for f in files if f[-7:] == '.params']
                if len(files) > 0:
                    files = sorted(files, reverse=True)  # put latest model first
                    model_name = files[0]
                    model.load_parameters(os.path.join('models', 'vision', 'experiments', FLAGS.backbone_from_id, model_name))
                    logging.info('Loaded backbone params: {}'.format(os.path.join('models', 'vision', 'experiments',
                                                                                  FLAGS.backbone_from_id, model_name)))

        if FLAGS.freeze_backbone and model is not None:
            for param in model.collect_params().values():
                param.grad_req = 'null'

        if FLAGS.temp_pool in ['gru', 'lstm']:
            model = CNNRNN(model, num_classes=len(test_set.classes), type=FLAGS.temp_pool, hidden_size=128)
        elif FLAGS.temp_pool in ['mean', 'max']:
            pass
        else:
            assert FLAGS.backbone == 'rdnet'  # ensure 3d net
            assert FLAGS.window in [8, 32]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.initialize()

    num_channels = 3
    if bool(FLAGS.flow):
        num_channels = 6
    if FLAGS.feats_model is None:
        if FLAGS.window == 1:
            logging.info(model.summary(mx.nd.ndarray.ones(shape=(1,
                                                                 num_channels, FLAGS.data_shape, FLAGS.data_shape))))
        else:
            logging.info(model.summary(mx.nd.ndarray.ones(shape=(1, FLAGS.window,
                                                                 num_channels, FLAGS.data_shape, FLAGS.data_shape))))
    else:
        if FLAGS.window == 1:
            logging.info(model.summary(mx.nd.ndarray.ones(shape=(1, 4096))))
        elif FLAGS.temp_pool not in ['max', 'mean']:
            logging.info(model.summary(mx.nd.ndarray.ones(shape=(1, FLAGS.window, 4096))))

    model.collect_params().reset_ctx(ctx)
    model.hybridize()

    if FLAGS.save_feats:
        best_score = -1
        best_epoch = -1
        with open(os.path.join('models', 'vision', 'experiments', FLAGS.model_id, 'scores.txt'), 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip().split() for line in lines]
            for ep, sc in lines:
                if float(sc) > best_score:
                    best_epoch = int(ep)
                    best_score = float(sc)

        logging.info('Testing best model from Epoch %d with score of %f' % (best_epoch, best_score))
        model.load_parameters(os.path.join('models', 'vision', 'experiments', FLAGS.model_id, "{:04d}.params".format(best_epoch)))
        logging.info('Loaded model params: {}'.format(
            os.path.join('models', 'vision', 'experiments', FLAGS.model_id, "{:04d}.params".format(best_epoch))))

        for data, sett in zip([train_data, val_data, test_data], [train_set, val_set, test_set]):
            save_features(model, data, sett, ctx)
        return

    start_epoch = 0
    if os.path.exists(os.path.join('models', 'vision', 'experiments', FLAGS.model_id)):
        files = os.listdir(os.path.join('models', 'vision', 'experiments', FLAGS.model_id))
        files = [f for f in files if f[-7:] == '.params']
        if len(files) > 0:
            files = sorted(files, reverse=True)  # put latest model first
            model_name = files[0]
            start_epoch = int(model_name.split('.')[0]) + 1
            model.load_parameters(os.path.join('models', 'vision', 'experiments', FLAGS.model_id, model_name), ctx=ctx)
            logging.info('Loaded model params: {}'.format(os.path.join('models', 'vision', 'experiments', FLAGS.model_id, model_name)))

    # Setup the optimiser
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': FLAGS.lr, 'momentum': FLAGS.momentum, 'wd': FLAGS.wd})

    # Setup Metric/s
    metrics = [Accuracy(label_names=test_set.classes),
               mx.metric.TopKAccuracy(5, label_names=test_set.classes),
               Accuracy(name='accuracy_no', label_names=test_set.classes[1:], ignore_labels=[0]),
               Accuracy(name='accuracy_o', label_names=test_set.classes[0],
                        ignore_labels=list(range(1, len(test_set.classes)))),
               PRF1(label_names=test_set.classes)]

    val_metrics = [Accuracy(label_names=test_set.classes),
                   mx.metric.TopKAccuracy(5, label_names=test_set.classes),
                   Accuracy(name='accuracy_no', label_names=test_set.classes[1:], ignore_labels=[0]),
                   Accuracy(name='accuracy_o', label_names=test_set.classes[0],
                            ignore_labels=list(range(1, len(test_set.classes)))),
                   PRF1(label_names=test_set.classes)]

    test_metrics = [Accuracy(label_names=test_set.classes),
                    mx.metric.TopKAccuracy(5, label_names=test_set.classes),
                    Accuracy(name='accuracy_no', label_names=test_set.classes[1:], ignore_labels=[0]),
                    Accuracy(name='accuracy_o', label_names=test_set.classes[0],
                             ignore_labels=list(range(1, len(test_set.classes)))),
                    PRF1(label_names=test_set.classes)]

    # Setup Loss/es
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    if FLAGS.temp_pool not in ['max', 'mean']:
        model = train_model(model, train_set, train_data, metrics, val_set, val_data, val_metrics, trainer, loss_fn, start_epoch, ctx, tb_sw)

    # model training complete, test it
    if FLAGS.temp_pool not in ['max', 'mean']:
        mod_path = os.path.join('models', 'vision', 'experiments', FLAGS.model_id)
    else:
        mod_path = os.path.join('models', 'vision', 'experiments', FLAGS.feats_model)
    best_score = -1
    best_epoch = -1
    with open(os.path.join(mod_path, 'scores.txt'), 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip().split() for line in lines]
        for ep, sc in lines:
            if float(sc) > best_score:
                best_epoch = int(ep)
                best_score = float(sc)

    logging.info('Testing best model from Epoch %d with score of %f' % (best_epoch, best_score))
    model.load_parameters(os.path.join(mod_path, "{:04d}.params".format(best_epoch)))
    logging.info('Loaded model params: {}'.format(os.path.join(mod_path, "{:04d}.params".format(best_epoch))))

    if FLAGS.temp_pool in ['max', 'mean']:
        assert FLAGS.backbone_from_id or FLAGS.feats_model  # if we doing temporal pooling ensure that we have loaded a pretrained net
        model = TemporalPooling(model, pool=FLAGS.temp_pool, num_classes=0, feats=FLAGS.feats_model!=None)

    tic = time.time()
    _ = test_model(model, test_data, test_set, test_metrics, ctx, vis=FLAGS.vis)

    if FLAGS.temp_pool not in ['max', 'mean']:
        str_ = 'Train set:'
        for i in range(len(train_set.classes)):
            str_ += '\n'
            for j in range(len(train_set.classes)):
                str_ += str(metrics[4].mat[i, j]) + '\t'
        print(str_)
    str_ = 'Test set:'
    for i in range(len(test_set.classes)):
        str_ += '\n'
        for j in range(len(test_set.classes)):
            str_ += str(test_metrics[4].mat[i, j]) + '\t'
    print(str_)

    str_ = '[Finished] '
    for metric in test_metrics:
        result = metric.get()
        if not isinstance(result, list):
            result = [result]
        for res in result:
            str_ += ', Test_{}={:.3f}'.format(res[0], res[1])
        metric.reset()

    str_ += '  # Samples: {}, Time Taken: {:.1f}'.format(len(test_set), time.time() - tic)
    logging.info(str_)

    # logging.info("Cleaning up, making test videos.")
    # for video in os.listdir(test_set.output_dir):
    #     frames_to_video(os.path.join(test_set.output_dir, video), os.path.join(test_set.output_dir, video[:-4]),
    #                     fps=int(25/FLAGS.every[2]))
    #     shutil.rmtree(os.path.join(test_set.output_dir, video))


def train_model(model, train_set, train_data, metrics, val_set, val_data, val_metrics, trainer, loss_fn, start_epoch, ctx, tb_sw=None):
    if FLAGS.epochs-start_epoch > 0:
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
            for metric in metrics:
                metric.reset()

            for i, batch in enumerate(train_data):  # loop over batches
                if FLAGS.max_batches > 0 and i > FLAGS.max_batches:
                    break
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
                        sum_losses.append(loss_fn(output, labels[ix]))

                    ag.backward(sum_losses)

                # step the optimizer
                trainer.step(FLAGS.batch_size)

                # store the epoch loss sums - avg loss across batch (avg across devices)
                train_sum_loss += sum([l.mean().asscalar() for l in sum_losses]) / len(sum_losses)

                # update metric
                for metric in metrics:
                    metric.update(labels, outputs)

                # logging
                if FLAGS.log_interval and not (i + 1) % FLAGS.log_interval:
                    str_ = '[Epoch {}][Batch {}/{}], LR: {:.2E}, Speed: {:.3f} samples/sec'.format(
                        epoch, i, num_batches, trainer.learning_rate, FLAGS.batch_size / (time.time() - btic))

                    str_ += ', {}={:.3f}'.format("loss:", train_sum_loss/(i*FLAGS.batch_size))
                    if tb_sw:
                        tb_sw.add_scalar(tag='Training_loss',
                                         scalar_value=train_sum_loss/(i*FLAGS.batch_size),
                                         global_step=(epoch * len(train_data) + i))
                    for metric in metrics:
                        result = metric.get()
                        if not isinstance(result, list):
                            result = [result]
                        for res in result:
                            str_ += ', {}={:.3f}'.format(res[0], res[1])
                            if tb_sw:
                                tb_sw.add_scalar(tag='Training_{}'.format(res[0]),
                                                 scalar_value=float(res[1]),
                                                 global_step=(epoch * len(train_data) + i))
                    logging.info(str_)

            # Format end of epoch logging string getting metrics along the way
            str_ = '[Epoch {}]'.format(epoch)

            for metric in metrics:
                result = metric.get()
                if not isinstance(result, list):
                    result = [result]
                for res in result:
                    str_ += ', Train_{}={:.3f}'.format(res[0], res[1])

            str_ += ', loss: {:.3f}'.format(train_sum_loss / len(train_data))

            vtic = time.time()
            _ = test_model(model, val_data, val_set, val_metrics, ctx)

            str_2 = 'Val set:'
            for i in range(len(train_set.classes)):
                str_2 += '\n'
                for j in range(len(train_set.classes)):
                    str_2 += str(val_metrics[4].mat[i, j]) + '\t'
            print(str_2)

            for metric in val_metrics:
                result = metric.get()
                if not isinstance(result, list):
                    result = [result]
                for res in result:
                    str_ += ', Val_{}={:.3f}'.format(res[0], res[1])
                    if tb_sw:
                        tb_sw.add_scalar(tag='Val_{}'.format(res[0]),
                                         scalar_value=float(res[1]),
                                         global_step=(epoch * len(train_data)))
                    if res[0] == 'AVG_NB_f1':
                        with open(os.path.join('models', 'vision', 'experiments', FLAGS.model_id, 'scores.txt'), 'a') as f:
                            f.write(str(epoch)+'\t'+str(float(res[1]))+'\n')

                metric.reset()

            str_ += ', Epoch Time: {:.1f}, Val Time: {:.1f}'.format(time.time() - tic, time.time() - vtic)

            logging.info(str_)

            model.save_parameters(os.path.join('models', 'vision', 'experiments', FLAGS.model_id, "{:04d}.params".format(epoch)))

    return model


# Testing/Validation function
def test_model(net, loader, dataset, metrics, ctx, vis=False):

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc='Testing'):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        idxs = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(x) for x in data]

        for metric in metrics:
            metric.update(labels, outputs)

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

    return metrics


def save_features(net, loader, dataset, ctx):
    for batch in tqdm(loader, desc='saving features', total=int(len(dataset)/FLAGS.batch_size)):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        # labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        idxs = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)

        for xi, x in enumerate(data):
            feat = net.backbone(x)
            feat = feat.asnumpy()
            idxsi = idxs[xi].asnumpy()
            for i in range(len(idxsi)):
                feat_path = dataset.save_feature_path(idxsi[i])
                if not os.path.exists(feat_path):
                    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
                    np.save(feat_path, feat[i])
                    print("Saving %s" % feat_path)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
