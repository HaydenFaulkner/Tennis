import cv2
import glob
import mxnet as mx
import numpy as np
import os
from scipy.misc import imresize
from tqdm import tqdm

from models.vision.flownet.model import get_flownet
from models.vision.flownet.utils import flow_to_image, crop, normalise


def process_two_images(model, files, ctx=None):
    """
    Process two images into one flow image
    Args:
        model: The model to use
        files: a list of 2 image paths
        ctx: the model ctx
    Returns:
    """
    if len(files) != 2:
        return None
    imgs = list()
    if isinstance(files[0], str) and isinstance(files[1], str) \
            and os.path.exists(files[0]) and os.path.exists(files[1]):
        imgs.append(cv2.cvtColor(cv2.imread(files[0]), cv2.COLOR_BGR2RGB))
        imgs.append(cv2.cvtColor(cv2.imread(files[1]), cv2.COLOR_BGR2RGB))
    else:
        return None, None

    imgs = crop(imgs)
    imgs = np.array(imgs)
    imgs = np.moveaxis(imgs, -1, 1)
    imgs = normalise(imgs)

    imgs = mx.nd.array(imgs, ctx=ctx)
    imgs = mx.nd.expand_dims(imgs, 0)  # add batch axis

    flow = model(imgs)  # run the model

    flow = flow.asnumpy()
    flow = flow.squeeze()
    flow = flow.transpose(1, 2, 0)
    img = flow_to_image(flow)
    img = imresize(img, 4.0)  # doing the bilinear interpolation on the img, NOT on flow cause was too hard :'(

    return img, flow


def process_imagedir(model, input_dir, output_dir=None, ctx=None):
    """
    Process a directory of images
    Args:
        model: The flownet model
        input_dir: The input image dir
        output_dir: The output image dir
        ctx: the model ctx
    Returns: output path of last saved sample
    """

    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        files = glob.glob(input_dir + "/**/*" + ext, recursive=True)
        if len(files) > 0:
            break

    if not len(files) > 0:
        print("Couldn't find any files in {}".format(input_dir))
        return None

    files.sort()

    for i in tqdm(range(1, len(files)), desc='Calculating Flow'):
        img, flow = process_two_images(model, files[i-1:i+1], ctx)
        dir, file = os.path.split(files[i])
        if int(file[:-4]) == 0:  # skip first frame of any video (assume numbered 0s)
            continue

        if output_dir is None:
            output_dir = 'flow'
        output_path = dir.replace(input_dir, output_dir)  # this keeps the recursive dir structure

        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, file), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return output_path


def process_video(model, input_path, output_path=None, ctx=None):
    """
    Process a video into a flow video
    Args:
        model: the flownet model
        input_path: the input path of the video
        output_path: the output path for the flow images
        ctx: the model ctx
    Returns: the output path
    """
    capture = cv2.VideoCapture(input_path)
    frames = []
    while_safety = 0
    while len(frames) < int(capture.get(cv2.CAP_PROP_FRAME_COUNT))-1:
        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        if image is None:
            while_safety += 1
            continue

        while_safety = 0  # reset the safety count
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    capture.release()

    if len(frames) < 2:
        return None

    if output_path is None:
        output_path = input_path[:-4] + '_flow.mp4'

    cropped_frames = crop(frames)
    h, w, _= cropped_frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (w, h))

    for i in tqdm(range(len(frames)-1), desc='Calculating Flow'):
        mx.nd.waitall()
        img, flow = process_two_images(model, frames[i:i+2], ctx)
        video.write(img)

    video.release()  # release the video

    return output_path


def generate_flows(image_dir, flow_dir):
    ctx = mx.gpu(0)
    net = get_flownet(ctx=ctx)
    net.hybridize()
    process_imagedir(net, input_dir=image_dir, output_dir=flow_dir, ctx=ctx)


if __name__ == '__main__':
    generate_flows(image_dir="data/frames", flow_dir="data/flow")

    # just for debugging
    # ctx = mx.gpu(0)
    #
    # net = get_flownet(ctx=ctx)
    # net.hybridize()
    # input_path = "/path/to/test.mp4"
    # process_video(net, input_path, ctx=ctx)
