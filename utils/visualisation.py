import cv2
import numpy as np
import os
import random
from tqdm import tqdm

from dataset import TennisSet

COLOURS = ((148, 148, 148),
           (255, 176, 143), (214, 127, 235), (143, 195, 255), (142, 235, 164), (255, 243, 140),
           (255, 214, 148), (235, 131, 154), (162, 147, 255), (145, 235, 223), (208, 255, 145))


def visualise_events(dataset, results, video_path, gt=None, max=-1):
    banner_height = 75
    gt_height = 0
    pred_border = 4
    cls_banner_height = 40

    if gt is not None:
        gt_height = 15

    classes = dataset.classes
    order = sorted(list(results.keys()))

    if max > 0:
        order = order[:min(len(order), max)]

    height, width, _ = cv2.imread(order[0]).shape

    cls_width = int(width/len(classes))

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (width, height+banner_height+cls_banner_height))

    canvas = np.zeros((height + banner_height + cls_banner_height, width, 3), dtype=np.uint8)

    for i, path in tqdm(enumerate(order), total=len(order), desc='Generating Vis'):
        assert os.path.exists(path)
        img = cv2.imread(path)

        canvas[:height, :, :] = img
        canvas[height:height + banner_height, :width-1, :] = canvas[height:height + banner_height, 1:, :]

        res = results[path]
        ind = np.argmax(res)

        prob = np.max(res)
        bar_height = int((banner_height-gt_height) * prob)

        canvas[height:height + banner_height, width-1:width, :] = (0, 0, 0)
        canvas[height + banner_height - bar_height - gt_height:height + banner_height - gt_height, width-1:width, :] = COLOURS[ind]

        if gt:
            gtp = gt[path]
            canvas[height+banner_height-int(gt_height*.9):height + banner_height - int(gt_height*.3), width - 1:width, :] = COLOURS[gtp]

        for c, cls in enumerate(classes):
            if ind == c:
                if gt is not None:
                    if ind == gtp:
                        canvas[height + banner_height + 2:, c * cls_width:(c + 1) * cls_width, :] = (0, 255, 0)
                    else:
                        canvas[height + banner_height + 2:, c * cls_width:(c + 1) * cls_width, :] = (0, 0, 255)
                else:
                    canvas[height + banner_height + 2:, c * cls_width:(c + 1) * cls_width, :] = (0, 0, 0)

                canvas[height + banner_height + 2+pred_border:-pred_border, c * cls_width + pred_border:(c + 1) * cls_width - pred_border, :] = COLOURS[c]
            else:
                canvas[height + banner_height + 2:, c * cls_width:(c + 1) * cls_width, :] = COLOURS[c]
            canvas = cv2.putText(canvas, cls, (int((c + 0.5) * cls_width - 22), height+banner_height+cls_banner_height-12),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        video.write(canvas)

    video.release()


if __name__ == '__main__':

    test_set = TennisSet(split='test', transform=None, every=1, padding=1, stride=1, window=1, model_id='0006',
                         split_id='02', balance=False, flow=False, feats_model=None, save_feats=False)

    results = dict()
    gts = dict()
    for i in range(len(test_set)):
        sample = test_set._samples[i]
        img_path = test_set.get_image_path(test_set._frames_dir, sample[0], sample[1])

        gt = test_set.classes.index(sample[2])

        fake_scores = list()
        for c in range(len(test_set.classes)):
            fake_scores.append(random.randint(1,20))

        results[img_path] = np.array([f/sum(fake_scores) for f in fake_scores])
        gts[img_path] = gt

    visualise_events(test_set, results, 'test.mp4', gt=gts)
