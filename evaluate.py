"""Evaluation script"""
from absl import app, flags
from absl.flags import FLAGS
import multiprocessing
import mxnet as mx
import numpy as np
import os
import pickle as pkl
import sys
import time
from tqdm import tqdm
import warnings

from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.accuracy import Accuracy

from models.vision.definitions import CNNRNN, FrameModel, TwoStreamModel, TemporalPooling
from dataset import TennisSet
from metrics.vision import PRF1
from models.vision.rdnet.r21d import get_r21d
from utils.visualisation import visualise_events

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
flags.DEFINE_string('split', 'test',
                    'the split to evaluate on either train, val, or test')
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
flags.DEFINE_integer('num_gpus', 1,
                     'Number of GPUs to use')
flags.DEFINE_integer('num_workers', -1,
                     'The number of workers should be picked so that it’s equal to number of cores on your machine '
                     'for max parallelization. If this number is bigger than your number of cores it will use up '
                     'a bunch of extra CPU memory. -1 is auto.')

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


def main(_argv):
    FLAGS.every = [int(s) for s in FLAGS.every]
    FLAGS.balance = [True if s.lower() == 'true' or s.lower() == 't' else False for s in FLAGS.balance]

    if FLAGS.num_workers < 0:
        FLAGS.num_workers = multiprocessing.cpu_count()

    ctx = [mx.gpu(i) for i in range(FLAGS.num_gpus)] if FLAGS.num_gpus > 0 else [mx.cpu()]

    key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
    print('\n'.join(f.serialize() for f in key_flags))

    # Data augmentation, will do in dataset incase window>1 and need to be applied image-wise
    transform_test = None
    if FLAGS.feats_model is None:
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

    test_set = TennisSet(split=FLAGS.split, transform=transform_test, every=FLAGS.every[2], padding=FLAGS.padding,
                         stride=FLAGS.stride, window=FLAGS.window, model_id=FLAGS.model_id, split_id=FLAGS.split_id,
                         balance=False, flow=bool(FLAGS.flow), feats_model=FLAGS.feats_model, save_feats=FLAGS.save_feats)

    print(test_set)

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
            model = TwoStreamModel(backbone_net, flow_net, len(test_set.classes))
        elif FLAGS.backbone == 'rdnet':
            model = FrameModel(backbone_net, len(test_set.classes), swap=True)
        else:
            model = FrameModel(backbone_net, len(test_set.classes))
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
                    print('Loaded backbone params: {}'.format(os.path.join('models', 'vision', 'experiments',
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
            print(model.summary(mx.nd.ndarray.ones(shape=(1, num_channels, FLAGS.data_shape, FLAGS.data_shape))))
        else:
            print(model.summary(mx.nd.ndarray.ones(shape=(1, FLAGS.window,
                                                          num_channels, FLAGS.data_shape, FLAGS.data_shape))))
    else:
        if FLAGS.window == 1:
            print(model.summary(mx.nd.ndarray.ones(shape=(1, 4096))))
        elif FLAGS.temp_pool not in ['max', 'mean']:
            print(model.summary(mx.nd.ndarray.ones(shape=(1, FLAGS.window, 4096))))

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

        print('Testing best model from Epoch %d with score of %f' % (best_epoch, best_score))
        model.load_parameters(os.path.join('models', 'vision', 'experiments', FLAGS.model_id, "{:04d}.params".format(best_epoch)))
        print('Loaded model params: {}'.format(
            os.path.join('models', 'vision', 'experiments', FLAGS.model_id, "{:04d}.params".format(best_epoch))))

        for data, sett in zip([test_data], [test_set]):
            save_features(model, data, sett, ctx)
        return

    if os.path.exists(os.path.join('models', 'vision', 'experiments', FLAGS.model_id)):
        files = os.listdir(os.path.join('models', 'vision', 'experiments', FLAGS.model_id))
        files = [f for f in files if f[-7:] == '.params']
        if len(files) > 0:
            files = sorted(files, reverse=True)  # put latest model first
            model_name = files[0]
            model.load_parameters(os.path.join('models', 'vision', 'experiments', FLAGS.model_id, model_name), ctx=ctx)
            print('Loaded model params: {}'.format(os.path.join('models', 'vision', 'experiments', FLAGS.model_id, model_name)))

    # Setup Metrics
    test_metrics = [Accuracy(label_names=test_set.classes),
                    mx.metric.TopKAccuracy(5, label_names=test_set.classes),
                    Accuracy(name='accuracy_no', label_names=test_set.classes[1:], ignore_labels=[0]),
                    Accuracy(name='accuracy_o', label_names=test_set.classes[0],
                             ignore_labels=list(range(1, len(test_set.classes)))),
                    PRF1(label_names=test_set.classes)]

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

    print('Testing best model from Epoch %d with score of %f' % (best_epoch, best_score))
    model.load_parameters(os.path.join(mod_path, "{:04d}.params".format(best_epoch)))
    print('Loaded model params: {}'.format(os.path.join(mod_path, "{:04d}.params".format(best_epoch))))

    if FLAGS.temp_pool in ['max', 'mean']:
        assert FLAGS.backbone_from_id or FLAGS.feats_model  # if we doing temporal pooling ensure that we have loaded a pretrained net
        model = TemporalPooling(model, pool=FLAGS.temp_pool, num_classes=0, feats=FLAGS.feats_model!=None)

    tic = time.time()

    results, gts = evaluate_model(model, test_data, test_set, test_metrics, ctx)

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
    print(str_)

    if FLAGS.vis:
        visualise_events(test_set, results, video_path=os.path.join('models', 'vision', 'experiments', FLAGS.model_id, 'results.mp4'), gt=gts)


# Testing/Validation function
def evaluate_model(net, loader, dataset, metrics, ctx):
    results = dict()
    ground_truths = dict()
    for batch in tqdm(loader, total=len(loader), desc='Evaluating'):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        idxs = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(x) for x in data]

        for metric in metrics:
            metric.update(labels, outputs)

        for di in range(len(outputs)):  # loop over devices
            idxs = [int(idx) for idx in idxs[di].asnumpy()]

            output = [o.asnumpy() for o in outputs[di]]
            if isinstance(outputs[0], list) or isinstance(outputs[0], tuple):
                for i in range(len(idxs)):  # loop over samples
                    sample = dataset._samples[idxs[i]]
                    img_path = dataset.get_image_path(dataset._frames_dir, sample[0], sample[1])
                    results[img_path] = [o[i] for o in output]
                    ground_truths[img_path] = dataset.classes.index(sample[2])
            else:
                for i in range(len(idxs)):  # loop over samples
                    sample = dataset._samples[idxs[i]]
                    img_path = dataset.get_image_path(dataset._frames_dir, sample[0], sample[1])
                    results[img_path] = output[i]
                    ground_truths[img_path] = dataset.classes.index(sample[2])

    return results, ground_truths


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
