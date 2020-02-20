"""Tennis Video Classification Dataset"""
from absl import app, flags, logging
import cv2
import math
import mxnet as mx
import numpy as np
import os
import random
import gluonnlp

from tqdm import tqdm

from utils.video import video_to_frames


class TennisSet:
    def __init__(self, root='data', captions=False, transform=None, split='train', every=1, balance=True, padding=1,
                 stride=1, window=1, model_id='0000', split_id='02', flow=False, max_cap_len=-1, vocab=None,
                 inference=False, feats_model=None, save_feats=False):
        self._root = root
        self._captions = captions
        self._split = split
        self._balance = balance
        self._every = every  # only get every nth frame from a video
        self._padding = padding  # temporal padding around event boundaries
        self._stride = stride  # temporal stride for frame sampling
        self._window = window  # input frame volume size =1:frames >1:clip, sample not frame based
        self._transform = transform
        self._flow = flow
        self._inference = inference
        self._save_feats = save_feats

        self._videos_dir = os.path.join(root, "videos")
        self._frames_dir = os.path.join(root, "frames")
        self._flow_dir = os.path.join(root, "flow")
        self._splits_dir = os.path.join(root, "splits")
        self._annotations_dir = os.path.join(root, "annotations")
        self._labels_dir = os.path.join(root, "annotations", "labels")
        self.output_dir = os.path.join(root, "outputs", model_id, split)
        self._load_feats = False
        self.feat_dir = os.path.join(root, "features", model_id)
        if feats_model is not None:
            self.feat_dir = os.path.join(root, "features", feats_model)
            self._load_feats = True

        self.classes = self._get_classes()

        self._samples, self._videos, self._events, self._points = self.load_data(split_id=split_id)

        self._video_lengths = self._get_video_lengths()

        if self._captions:
            self._samples = list(self._points.keys())

            caps = [p[4] for p in self._points.values()]
            words = ' '.join(caps).split()
            if vocab is None:
                counter = gluonnlp.data.count_tokens(words)
                self.vocab = gluonnlp.Vocab(counter)
            else:
                self.vocab = vocab

            for i in range(len(self)):
                point_id = self._samples[i]
                cap = self._points[point_id][4]
                # For max_cap_len < 0, we do not clip the sequence
                if max_cap_len >= 0:
                    cap_ids = self.vocab[cap.split()[:max_cap_len]]
                else:
                    cap_ids = self.vocab[cap.split()]
                cap_ids.insert(0, self.vocab[self.vocab.bos_token])
                cap_ids.append(self.vocab[self.vocab.eos_token])
                cap_ids = np.array(cap_ids, dtype=np.int32)
                self._points[point_id].append(cap_ids)

        elif self._balance:
            self._samples = self._balance_classes()

    def get_captions(self, ids=False, split=False):
        caps = list()
        for i in range(len(self)):
            point_id = self._samples[i]
            if ids:
                cap = self._points[point_id][5]
            else:
                cap = self._points[point_id][4]

            if split:
                caps.append(cap.split())
            else:
                caps.append(cap)

        return caps

    def __str__(self):
        return '\n\n' + self.__class__.__name__ + '\n' + self.stats() + '\n'

    def stats(self):
        """
        Get a stats string for the dataset

        Returns:
            str: stats string
        """
        output = ''
        output += 'Split: {}\n'.format(self._split)

        if self._captions:
            output += '{0: <8} {1: <8} {2: <5}\n'.format('# Points', '# Frames', 'FperP')
            points = 0
            frames = 0
            for s in self._samples:
                points += 1
                frames += int(self._points[s][2]) - int(self._points[s][1])
            output += '{0: <8} {1: <8} {2: <5}\n'.format(points, frames, int(frames/points))

        else:
            classes = self.classes
            frame_counts = [0]*len(classes)
            for s in self._samples:
                frame_counts[classes.index(s[2])] += 1
            event_counts = [0]*len(classes)
            for e in self._events:
                event_counts[classes.index(e[3])] += 1

            output += '{0: <6} {1: <8} {2: <8} {3: <5}\n'.format('Class', '# Frames', '# Events', 'FperE')
            for i, c in enumerate(classes):
                output += '{0: <6} {1: <8} {2: <8} {3: <5}\n'.format(c, frame_counts[i], event_counts[i],
                                                                     int(frame_counts[i]/(event_counts[i]+.00001)))
        return output

    def __len__(self):
        return len(self._samples)

    @staticmethod
    def get_image_path(root_dir, video_name, frame_number, chunk_size=1000):
        chunk = int(frame_number/chunk_size)*chunk_size
        return os.path.join(root_dir, video_name+'.mp4', '{:010d}'.format(chunk), '{:010d}.jpg'.format(frame_number))

    @staticmethod
    def get_feature_path(feat_dir, video_name, frame_number, chunk_size=1000):
        chunk = int(frame_number / chunk_size) * chunk_size
        return os.path.join(feat_dir, video_name + '.mp4', '{:010d}'.format(chunk), '{:010d}.npy'.format(frame_number))

    def save_feature_path(self, idx, chunk_size=1000):
        sample = self._samples[idx]
        video_name = sample[0]
        frame_number = sample[1]
        chunk = int(frame_number / chunk_size) * chunk_size
        return os.path.join(self.feat_dir, video_name + '.mp4', '{:010d}'.format(chunk), '{:010d}.npy'.format(frame_number))

    def __getitem__(self, idx):
        sample = self._samples[idx]
        if self._captions:
            point = self._points[sample]
            vid = point[0]
            start = int(point[1])
            end = int(point[2])
            cap = point[5]

            imgs = list()
            c = 0
            for f in range(start, end):
                if c % self._every == 0:
                    # print('%d/%d'%(f-start, end-start))
                    if self._load_feats:
                        feats_path = self.get_feature_path(self.feat_dir, vid, f)
                        img = mx.nd.array(np.load(feats_path))
                    else:
                        img_path = self.get_image_path(self._frames_dir, vid, f)
                        img = mx.image.imread(img_path, 1)

                    if self._transform is not None:
                        img = self._transform(img)
                    imgs.append(img)
                c += 1

            imgs = mx.nd.stack(*imgs)

            if self._inference:
                return imgs, cap, len(imgs), len(cap), idx
            else:
                return imgs, cap, len(imgs), len(cap)
        else:

            img_path = self.get_image_path(self._frames_dir, sample[0], sample[1])
            flw_path = self.get_image_path(self._flow_dir, sample[0], sample[1])
            label = self.classes.index(sample[2])

            if self._window > 1:
                imgs = list()
                window_offsets = list(range(int(-self._window/2), int(math.ceil(self._window/2))))
                window_offsets = window_offsets[:]
                for offset in window_offsets:
                    # need to get max frame for video, has to be an 'every' frame
                    max_frame = self._video_lengths[sample[0]]-self._every
                    for i in range(self._every):
                        if (max_frame - i) % self._every == 0:
                            max_frame -= i
                            break
                    frame = min(max(0, sample[1]+offset*self._stride), int(max_frame))  # bound the frame
                    if self._load_feats:
                        feats_path = self.get_feature_path(self.feat_dir, sample[0], frame)
                        img = mx.nd.array(np.load(feats_path))
                    else:
                        img_path = self.get_image_path(self._frames_dir, sample[0], frame)
                        img = mx.image.imread(img_path, 1)

                    if self._flow:
                        flw_path = self.get_image_path(self._flow_dir, sample[0], frame)
                        flw = mx.image.imread(flw_path, 1)
                        img = mx.nd.concatenate([img[8:-8][:][:], flw], axis=-1)

                    if self._transform is not None:
                        img = self._transform(img)
                    imgs.append(img)
                img = mx.nd.stack(*imgs)
            else:

                if self._load_feats:
                    feats_path = self.get_feature_path(self.feat_dir, sample[0], sample[1])
                    img = mx.nd.array(np.load(feats_path))
                else:
                    img = mx.image.imread(img_path, 1)

                if self._flow:
                    flw = mx.image.imread(flw_path, 1)
                    img = mx.nd.concat(img[8:-8][:][:], flw, dim=-1)

                if self._transform is not None:
                    img = self._transform(img)

            return img, label, idx

    def get_data_lens(self):
        assert self._captions
        lens = list()
        for i in range(len(self)):
            sample = self._samples[i]

            point = self._points[sample]
            start = int(point[1])
            end = int(point[2])
            cap = point[5]

            lens.append((int((end-start+1)/self._every), len(cap)))
        return lens

    @staticmethod
    def _get_classes():
        """
        Gets a list of class names as specified in the imagenetvid.names file

        Returns:
            list : a list of strings

        """
        names_file = os.path.join('data', 'classes.names')
        with open(names_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)

    def _balance_classes(self):
        """
        Balance the dataset on 'Other' class, with next most sampled class, uses uniform random sampling

        Returns:
            list: the balanced set of samples
        """
        #
        counts = self.class_counts()
        next_most = max(counts[1:])
        ratio = next_most/float(counts[0]+1)

        balanced = list()
        for sample in self._samples:
            if sample[2] == 'OTH' and random.uniform(0, 1) > ratio:
                continue
            balanced.append(sample)
        samples = balanced

        return samples

    def class_counts(self):
        """
        Get the sample counts for each class

        Returns:
            list: of ints with length of classes with the sample counts per class
        """
        classes = self.classes
        counts = [0]*len(classes)
        for s in self._samples:
            counts[classes.index(s[2])] += 1  # todo assumes frames at the moment
        return counts

    def load_data(self, split_id='01'):
        """
        Load the data

        Args:
            split_id (str): the split id either '01' or '02'

        Returns:
            list: of samples [[video, frame, class], ...]
            list: of videos [video1, video2, ...]
            list: of events [[video, start_frame, last_frame, cur_class], ...]
        """
        splits_file = os.path.join(self._splits_dir, split_id, self._split + '.txt')

        # load the splits file
        if os.path.exists(splits_file):
            logging.info("Loading data from {}".format(splits_file))
            with open(os.path.join(self._splits_dir, split_id, self._split + '.txt'), 'r') as f:
                lines = f.readlines()
                samples = [[line.rstrip().split()[0], int(line.rstrip().split()[1])] for line in lines]

            # make a list of the videos
            videos = set()
            for s in samples:
                videos.add(s[0])
            videos = list(videos)

            labels = dict()
            for video in videos:
                labels[video] = dict()

            if self._save_feats:
                for v in videos:
                    min_f = 100000000
                    max_f = 0
                    for s in samples:
                        if s[0] == v:
                            min_f = min(min_f, s[1])
                            max_f = max(max_f, s[1])
                    for i in range(1, 256):
                        samples.append([v, min_f-i])
                        samples.append([v, max_f+i])
                        labels[v][min_f-i] = 'OTH'
                        labels[v][max_f+i] = 'OTH'

            # verify images exist, if not try and extract, if not again then ignore
            for i in range(2):  # go around twice, so if not all samples found extract, then re-check
                samples_exist = list()
                samples_exist_flag = True

                for s in samples:
                    if not os.path.exists(self.get_image_path(self._frames_dir, s[0], s[1])):
                        if i == 0:  # first attempt checking all samples exist, try extracting
                            samples_exist_flag = False  # will flag to extract frames

                            logging.info("{} does not exist, will extract frames."
                                         "".format(self.get_image_path(self._frames_dir, s[0], s[1])))
                            break

                        else:  # second attempt, just ignore samples
                            logging.info("{} does not exist, will ignore sample."
                                         "".format(self.get_image_path(self._frames_dir, s[0], s[1])))
                    else:
                        samples_exist.append(s)

                if samples_exist_flag:  # all samples exist
                    break
                else:
                    for video in videos:  # lets extract frames
                        video_to_frames(video_path=os.path.join(self._videos_dir, video + '.mp4'),  # assuming .mp4
                                        frames_dir=self._frames_dir,
                                        chunk_size=1000)

            samples = samples_exist

            # load the class labels for each sample
            for video in videos:
                with open(os.path.join(self._labels_dir, video + '.txt'), 'r') as f:
                    lines = f.readlines()
                    lines = [l.rstrip().split() for l in lines]

                for line in lines:
                    labels[video][int(line[0])] = line[1]

            # a dict of the frames in the set for each video
            in_set = dict()
            for video in videos:
                in_set[video] = list()

            #  add class labels to each sample
            for i in range(len(samples)):
                samples[i].append(labels[samples[i][0]][samples[i][1]])
                in_set[samples[i][0]].append(samples[i][1])

            # load events (consecutive frames with same class label)
            events = list()
            for video in in_set.keys():
                cur_class = 'OTH'
                start_frame = -1
                for frame in sorted(in_set[video]):
                    if start_frame < 0:
                        start_frame = frame
                        last_frame = frame
                    if labels[video][frame] != cur_class:
                        events.append([video, start_frame, last_frame, cur_class])
                        cur_class = labels[video][frame]
                        start_frame = frame
                    last_frame = frame

                events.append([video, start_frame, last_frame, cur_class])  # add the last event

            # let's make up the points data
            with open(os.path.join(self._annotations_dir, 'points.txt'), 'r') as f:
                lines = f.readlines()
            points = [l.rstrip().split() for l in lines]

            # add caps
            with open(os.path.join(self._annotations_dir, 'captions.txt'), 'r') as f:
                lines = f.readlines()
                lines = [l.rstrip().split('\t') for l in lines]
            caps = dict()
            for l in lines:
                caps[l[0]] = l[1]
            for i in range(len(points)):
                points[i].append(caps[points[i][0]])

            # filter out points not in this split
            points_dict = dict()
            for point in points:
                if point[1] in videos and int(point[2]) in in_set[point[1]]:
                    points_dict[point[0]] = point[1:]

            return samples, videos, events, points_dict
        else:
            logging.info("Split {} does not exist, please make sure it exists to load a dataset.".format(splits_file))
            return None, None, None

    def _get_video_lengths(self):
        """
        get the video lengths

        :return: the video lengths dictionary
        """
        lengths = dict()
        for sample in self._samples:
            video_name = sample[0]
            if video_name not in lengths:
                largest_dir = sorted(os.listdir(os.path.join(self._frames_dir, video_name + '.mp4')))[-1]
                assert largest_dir.isdigit(), "Expects the directory {} to only contain numbered subdirs".format(
                    os.path.join(self._frames_dir, video_name + '.mp4'))
                largest_file = sorted(os.listdir(os.path.join(self._frames_dir, video_name + '.mp4', largest_dir)))[-1]
                lengths[video_name] = int(largest_file[:-4])

        return lengths

    def save_sample(self, idx, outputs=None):  # todo
        sample = self._samples[idx]
        img_path = self.get_image_path(self._frames_dir, sample[0], sample[1])
        save_img_path = self.get_image_path(self.output_dir, sample[0], sample[1])

        img = cv2.imread(img_path)

        # for i, l in enumerate(label):
        #     img[-20:, i*50:(i+1)*50, :] = 255*l
        #
        # if outputs is not None:
        #     if isinstance(outputs, list):
        #         logits = [int(np.argmax(o)) for o in outputs]  # multiple outputs
        #     else:
        #         logits = [int(np.argmax(outputs))]  # single output
        #
        # for i, l in enumerate(logits):
        #     img[-40:-20, i*50:(i+1)*50, :] = 255*l

        # Save the extracted image
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
        cv2.imwrite(save_img_path, img)

    def calc_flow_mean_std(self, every=100):
        assert self._flow
        m0, m1, m2, s0, s1, s2, c = 0, 0, 0, 0, 0, 0, 0
        for i in tqdm(range(len(self))):
            if i % every == 0:
                s = self.__getitem__(i)
                s = s[0][:, :, 3:].asnumpy()
                m0 += s[:, :, 0].mean() / 256
                m1 += s[:, :, 1].mean() / 256
                m2 += s[:, :, 2].mean() / 256
                s0 += s[:, :, 0].std() / 256
                s1 += s[:, :, 1].std() / 256
                s2 += s[:, :, 2].std() / 256
                c += 1
        return m0/c, m1/c, m2/c, s0/c, s1/c, s2/c


def main(_argv):

    ts = TennisSet(split='val', captions=True, balance=False, split_id='01', flow=True)
    for s in tqdm(ts):
        pass
    print(ts.stats())

    ts = TennisSet(split='val', balance=False, split_id='01')
    print(ts.stats())

    ts = TennisSet(split='test', balance=False, split_id='01')
    print(ts.stats())

    ts = TennisSet(split='train', balance=False, split_id='02')
    print(ts.stats())

    ts = TennisSet(split='val', balance=False, split_id='02')
    print(ts.stats())

    ts = TennisSet(split='test', balance=False, split_id='02')
    print(ts.stats())


if __name__ == '__main__':

    try:
        app.run(main)
    except SystemExit:
        pass

