"""
Takes a directory containing a sequence of frames, and turns them into a video

At the moment only works for 3 channel images
"""
import os
import numpy as np
import argparse
import cv2


def frames_2_video(frames_dir, save_path, fps=25):

    # Make sure it is an .avi file
    # TODO: Is this the only format OpenCV can handle... may cause issues on other peoples systems
    if save_path[:-4] != '.avi':
        save_path += '.avi'

    if os.path.isfile(save_path):
        try:
            os.path.remove(save_path)
        except OSError:
            pass

    files = [f for f in os.path.listdir(frames_dir) if f.endswith('.png') and os.path.isfile(os.path.join(frames_dir,f))]

    if len(files) > 0:
        # get first file to check frame size
        image = cv2.imread(os.path.join(frames_dir, files[0]))
        width = np.shape(image)[0]
        height = np.shape(image)[1]

        files.sort()

        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), fps, (width,height))

        for filename in files:
            print("Doing %s" % filename)
            image = cv2.imread(os.path.join(frames_dir, filename))
            video.write(image)

        video.release()
    else:
        print("Couldn't find any files in %s" % frames_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--frames_dir", type=str, required=True,
                    help='Path to directory containing video files, or singular video file path')
    parser.add_argument("-o", "--save_path", type=str, required=True,
                    help='Path of directory where to save images .png')
    parser.add_argument("-fps", "--fps", type=int, default=25,
                        help='Framerate to save video as. If omitted will use 25 fps.')

    args = parser.parse_args()

    frames_2_video(frames_dir=args.frames_dir,
                   save_path=args.save_path,
                   fps=args.fps)
