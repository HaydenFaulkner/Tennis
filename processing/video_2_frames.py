"""
video_2_frames.py extracts and saves frames from a video individually into .png files
"""


import numpy as np
import os
import argparse
import cv2

from config import config


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def load_slice_txt(slice_path):
    if not os.path.isfile(slice_path):
        print("Slice file doesn't exist: " + slice_path)
        return None
    else:
        with open(slice_path, 'r') as f:
            slices = [[num(slice.split()[0]), num(slice.split()[1])] for slice in f.readlines()]
        return slices


def crop_n_squash(frame, crop, squash):
    # Crop Frame
    if crop[0] is None and crop[1] is None:
        # print 'No crop size specified, using original frame size.'
        frame2 = frame
    elif crop[0] is None:
        # print 'No crop width specified, using original frame width.'
        frame2 = frame[int((np.shape(frame)[0]-crop[1])/2.0):crop[1]+int((np.shape(frame)[0]-crop[1])/2.0), :, :]
    elif crop[1] is None:
        # print 'No crop height specified, using original frame height.'
        frame2 = frame[:, int((np.shape(frame)[1]-crop[0])/2.0):crop[0]+int((np.shape(frame)[1]-crop[0])/2.0), :]
    else:
        frame2 = frame[int((np.shape(frame)[0]-crop[1])/2.0):crop[1]+int((np.shape(frame)[0]-crop[1])/2.0),
                 int((np.shape(frame)[1]-crop[0])/2.0):crop[0]+int((np.shape(frame)[1]-crop[0])/2.0),
                 :]

    # Squash / Resize Frame
    if squash[0] is None and squash[1] is None:
        # print 'No squash size specified, using cropped frame size.'
        frame = frame2
    elif squash[0] is None:
        # print 'No squash width specified, using cropped frame width.'
        frame = cv2.resize(frame2, (np.shape(frame2)[1], squash[1]))
    elif squash[1] is None:
        # print 'No squash height specified, using cropped frame height.'
        frame = cv2.resize(frame2, (squash[0], np.shape(frame2)[2]))
    else:
        frame = cv2.resize(frame2, (squash[0], squash[1]))

    return frame


def video_2_frames(video_path, save_path, slices=None, fps=None, crop=None, squash=None):

    if not os.path.isfile(video_path):
        print("Can't interpret the provided video path")
        return None

    capture = cv2.VideoCapture(video_path)
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 1:
        print("Video file can't be loaded: " + video_path)
        print("This may be related to OpenCV and FFMPEG")
        return None

    frames = []
    if slices is not None:
        for slice in slices:
            if isinstance(slice[0], float): # determine whether slices are times (float) or frames (int)
                # if slice is timestamp, need to convert into frames
                frames.extend(range(int(round(slice[0] * video_fps)), int(round(slice[1] * video_fps))))
            else:
                frames.extend(range(slice[0], slice[1]))
    else:
        frames.extend(range(0, frame_count))

    if fps is not None:
        fps = min(video_fps, fps)  # ensure don't sample more frames that exist
        print("Sampling %d frames per second" % fps)
        frames_tmp = []
        for f in frames:
            if f % int(round(video_fps / float(fps))) == 0:
                frames_tmp.append(f)
        frames = frames_tmp

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    current = 0
    end = max(frames)

    for run in range(10):
        while True:
            flag, frame = capture.read()
            if flag == 0:
                break
            if current > end:
                break

            if current in frames:
                frame = crop_n_squash(frame, crop=crop, squash=squash)

                cv2.imwrite("%s/%08d.png" % (save_path, current), frame)

            current += 1

        # Perform check to see if all frames written out
        # if not redo, up to a max of 10 times if still not done all list them with error
        missing_frames = []
        for f in frames:
            if not os.path.isfile("%s/%08d.png" % (save_path, f)):
                missing_frames.extend(f)

        if len(missing_frames) < 1:
            break

        frames = missing_frames

        if run == 10-1:
            if len(missing_frames) < 20:
                print("Was unable to save the following frames:")
                print(missing_frames)
            else:
                print("Was unable to save %d frames. See %s/missing_frames.txt" % (len(missing_frames), save_path))

            with open("%s/missing_frames.txt" % save_path, "w") as file:
                for f in missing_frames:
                    file.write("%d\n" % f)
            break


def videos_2_frames(videos_dir, save_dir, slices_dir, fps=None, crop=None, squash=None):
    # Given directory
    if os.path.isdir(videos_dir):
        videos = os.listdir(videos_dir)
        for v in range(len(videos)):
            print("------------\n%d / %d\n-------------" % (v, len(videos)))
            video_id = videos[v]
            # video_id = video_id[video_id.rfind('.'):]  # remove extension if there is one
            video_2_frames(video_path=os.path.join(videos_dir, videos[v]),
                           save_path=os.path.join(save_dir, video_id),
                           slices=load_slice_txt(os.path.join(slices_dir, video_id)),
                           fps=fps,
                           crop=crop,
                           squash=squash)
    # Given single file
    elif os.path.isfile(videos_dir):
        video_id = videos_dir.split("/")[-1]
        # video_id = video_id[video_id.rfind('.'):]  # remove extension if there is one
        video_2_frames(video_path=videos_dir,
                       save_path=os.path.join(save_dir, video_id),
                       slices=load_slice_txt(os.path.join(slices_dir, video_id)),
                       fps=fps,
                       crop=crop,
                       squash=squash)
    else:
        print("Can't interpret the provided video path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--videos_dir", type=str, default=config.directories.videos,
                    help='Path to directory containing video files, or singular video file path')
    parser.add_argument("-o", "--save_dir", type=str, default=config.directories.frames,
                    help='Path of directory where to save images .png')
    parser.add_argument("-s", "--slices_dir", type=str, default=config.directories.slices,
                        help='Path to a slices dir. If included will only do sliced frames. If no slice files found will do all frames.')
    parser.add_argument("-fps", "--fps", type=int, default=config.frames.fps,
                        help='Frames to sample (extract) per second. If omitted will used config settings. If None will extract every frame.')

    parser.add_argument("-cw", "--crop_width", type=int, default=config.frames.crop[0],
                    help='Will crop frame to this width around center BEFORE squashing. If omitted will use config settings. If None will use frame width.')
    parser.add_argument("-ch", "--crop_height", type=int, default=config.frames.crop[1],
                    help='Will crop frame to this height around center BEFORE squashing. If omitted will use config settings. If None will use frame height.')
    parser.add_argument("-sw", "--squash_width", type=int, default=config.frames.squash[0],
                    help='Output frame width. Will squash to this width around center. If omitted will use config settings. If None will use crop width.')
    parser.add_argument("-sh", "--squash_height", type=int, default=config.frames.squash[1],
                    help='Output frame height. Will squash to this height around center. If omitted will use config settings. If None will use crop height.')
    args = parser.parse_args()

    videos_2_frames(videos_dir=args.videos_dir,
                    save_dir=args.save_dir,
                    slices_dir=args.slices_dir,
                    fps=args.fps,
                    crop=[args.crop_width, args.crop_height],
                    squash=[args.squash_width, args.squash_height])


