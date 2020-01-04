import cv2
import numpy as np
import _pickle as pkl

import mxnet as mx

def convert_weights(model, load_path, n_classes, n_layers=34, save_path=None, dataset='sports1m'):
    """
    Used to convert model weights from official .pkl to gluon .params

    Args:
        model: the gluon model
        load_path: the path to the pickle
        n_layers: the number of layers in this model, can only be 34 or 152
        save_path: the path to save the .params, if None will put in same place as load_path
        dataset: the dataset the model was changed on, this affects the number of classes it outputs

    Returns: the save_path

    """
    # load from .pkl from github.com/facebookresearch/VMZ/blob/master/tutorials/model_zoo.md
    assert load_path[-4:] == '.pkl', 'Must be a .pkl file'
    # only 34 and 152 layer nets avail
    assert n_layers in [34, 152], 'Can only be a 34 or 152 layer model'

    if save_path is None:
        save_path = load_path[:-4] + ".params"

    with open(load_path, 'rb') as f:
        x = pkl.load(f, encoding='latin1')

    net_layers = {18: [2, 2, 2, 2],
                  34: [3, 4, 6, 3],
                  50: [3, 4, 6, 3],
                  101: [3, 4, 23, 3],
                  152: [3, 8, 36, 3]}

    r = list()
    for stage, blocks in enumerate(net_layers[n_layers]):
        for block in range(blocks):
            r.append(['comp_%d_' % len(r), 'r21dv10_stage%d_block%d_' % (stage + 1, block + 1)])

    r += [['conv1_', 'r21dv10_init_'],
          ['_w', '_weight'],
          ['_rm', '_running_mean'],
          ['_riv', '_running_var'],
          ['_spatbn', '_conv'],
          ['last_out_L' + str(n_classes) + '_', 'r21dv10_dense0_']]
    if n_layers == 34:
        r += [['shortcut_projection_3_conv_', 'r21dv10_stage2_block1_down_'],
              ['shortcut_projection_7_conv_', 'r21dv10_stage3_block1_down_'],
              ['shortcut_projection_13_conv_', 'r21dv10_stage4_block1_down_'],
              ['shortcut_projection_3_', 'r21dv10_stage2_block1_down_'],
              ['shortcut_projection_7_', 'r21dv10_stage3_block1_down_'],
              ['shortcut_projection_13_', 'r21dv10_stage4_block1_down_']]
    elif n_layers == 152:
        r += [['shortcut_projection_0_conv_', 'r21dv10_stage1_block1_down_'],
              ['shortcut_projection_3_conv_', 'r21dv10_stage2_block1_down_'],
              ['shortcut_projection_11_conv_', 'r21dv10_stage3_block1_down_'],
              ['shortcut_projection_47_conv_', 'r21dv10_stage4_block1_down_'],
              ['shortcut_projection_0_', 'r21dv10_stage1_block1_down_'],
              ['shortcut_projection_3_', 'r21dv10_stage2_block1_down_'],
              ['shortcut_projection_11_', 'r21dv10_stage3_block1_down_'],
              ['shortcut_projection_47_', 'r21dv10_stage4_block1_down_']]

    r += [['_conv_relu_', '_'],
          ['_dense0_beta', '_dense0_bias']
          ]

    keys = x['blobs'].keys()
    comp = dict()
    for k in keys:
        kk = k
        if kk[-2:] == '_s':
            kk = kk[:-2] + '_gamma'
        if kk[-2:] == '_b':
            kk = kk[:-2] + '_beta'
        for rep in r:
            kk = kk.replace(rep[0], rep[1])
        comp[kk] = k

    found = []
    not_found = []
    pdict = model.collect_params()
    for k in pdict.keys():
        if k in comp:
            print(k + ' :: '+str(pdict[k].data().shape) + ' <--loaded from-- '+
                  comp[k]+' :: '+str(x['blobs'][comp[k]].shape))
            found.append(k)
        else:
            not_found.append(k + ' :: ' + str(pdict[k].data().shape))

    error = False
    print("Parameters from mxnet model not matched:")
    for k in not_found:
        error = True
        print(k)
    print("Parameters from pickle model not matched:")
    for k in sorted(comp):
        if k not in found:
            error = True
            print(k + " :: " + str(x['blobs'][comp[k]].shape))

    if error:
        print("The model and save params don't align, can't load.")
        return None

    # put in the params
    param = model.collect_params()
    for k in pdict.keys():
        param._params[k]._data[0] = mx.nd.array(x['blobs'][comp[k]])

    # save out the params mx style
    model.save_parameters(save_path)

    return save_path

def get_test_frames(video_path, length_rgb=8):
    """
    Used to get a single sample of frames to test on

    Args:
        video_path: path to test video

    Returns: mx.nd.array of frames

    """
    # load in the video frames
    capture = cv2.VideoCapture(video_path)
    frame = 0
    frames = []
    while frame < length_rgb:
        ret, image = capture.read()
        if ret == 0 or image is None:
            continue
        if frame < length_rgb:
            frames.append(image)
        frame += 1
    capture.release()

    return transform_frames(frames, length_rgb=length_rgb)  # do transforms and return


def transform_frames(frames, length_rgb=8, scale_h=128, scale_w=171, crop=112, bgr=True):
    """
    Used to get a single sample of frames to test on

    Args:
        frames: list of frames
        length_rgb: clip length for net input (video needs to be at least this long)
        scale_h: scale the frames to this height
        scale_w: scale the frames to this width
        crop: crop to network input
        bgr: are these frames bgr?

    Returns: mx.nd.array of frames
    """

    # calculate offsets for centre cropping
    off_w = int((scale_w - crop)/2)
    off_h = int((scale_h - crop)/2)

    # per frame transforms
    for i in range(length_rgb):
        frames[i] = frames[i] / 255 # put values in range [0,1]
        frames[i] = cv2.resize(frames[i], (scale_w, scale_h), fx=0, fy=0, interpolation=cv2.INTER_AREA) # resize
        if bgr:
            frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        frames[i] = frames[i][off_h:off_h+crop,off_w:off_w+crop,:] # centre crop : t,h,w,c

    frames = np.array(frames)
    frames = np.moveaxis(frames, -1, 1)  # could do these in mx but meh oh well, t,h,w,c -> t,c,h,w
    frames = np.expand_dims(frames, 0)  # t,c,h,w -> b,t,c,h,w
    frames = mx.nd.array(frames)  # make mx nd array

    # normalise with the mean and std dev of kinetics
    for i in range(length_rgb):
        frames[0][i] = mx.nd.image.normalize(frames[0][i],
                                             mean=[0.43216, 0.394666, 0.37645],
                                             std=[0.22803, 0.22145, 0.216989])

    frames = mx.nd.swapaxes(frames, dim1=1, dim2=2)  # b,t,c,h,w -> b,c,t,h,w

    return frames