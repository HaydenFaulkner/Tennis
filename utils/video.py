"""
Video utilities such as converting between videos and frames and vice versa
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import glob
import multiprocessing
import os
import sys
from tqdm import tqdm


def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar

    Args:
        iteration (int): current iteration
        total (int): total iterations
        prefix (str): prefix string (default is '')
        suffix (str): suffix string (default is '')
        decimals (int): positive number of decimals in percent complete (default is 3)
        bar_length (int): character length of bar (default is 100)

    Returns:
        None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
    sys.stdout.flush()  # flush to stdout


def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture

    Args:
        video_path (str): path of the video
        frames_dir (str): the directory to save the frames
        overwrite (bool): to overwrite frames that already exist? (default is False)
        start (int): start frame (default is -1)
        end (int): end frame (default is -1)
        every (int): frame spacing (default is 1)

    Returns:
        int: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        ret, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if ret == 0 or image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            # save in start of chunk subdirectory in video name subdirectory
            save_path = os.path.join(frames_dir, video_filename, "{:010d}".format(start), "{:010d}.jpg".format(frame))
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=False, every=1, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing

    Args:
        video_path (str): path to the video
        frames_dir (str): directory to save the frames
        overwrite (bool): overwrite frames if they exist? (default is False)
        every (int): extract every this many frames (default is 1)
        chunk_size (int): how many frames to split into chunks (one chunk per cpu core process) (default is 1000)

    Returns:
        str: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation, can't read videos!!!\n"
              "You may need to install OpenCV by source not pip")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame

    for frame_chunk in frame_chunks:
        # make directory to save frames, its a sub dir in the frames_dir with the video name
        # also since file systems hate lots of files in one directory, lets put separate chunks in separate directories
        os.makedirs(os.path.join(frames_dir, video_filename, "{:010d}".format(frame_chunk[0])), exist_ok=True)

    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            print_progress(i, len(frame_chunks)-1, prefix=prefix_str, suffix='Complete')  # print it's progress

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames


def frames_to_video(frames_dir, video_path, fps=30):
    """
    Generates a .mp4 video from a directory of frames

    Args:
        frames_dir (str): the directory containing the frames
                          note that this and any subdirs be looked through recursively
        video_path (str): path to save the video
        fps (int): the frames per second to make the output video (default is 30)

    Returns:
        str: the output video path, or None if error
    """

    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible
    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible

    # add the .mp4 extension if it isn't already there
    if video_path[-4:] != ".mp4":
        video_path += ".mp4"

    # get the frame file paths
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        files = glob.glob(frames_dir + "/**/*" + ext, recursive=True)
        if len(files) > 0:
            break

    # couldn't find any images
    if not len(files) > 0:
        print("Couldn't find any files in {}".format(frames_dir))
        return None

    # get first file to check frame size
    image = cv2.imread(files[0])
    height, width, _ = image.shape  # need to get the shape of the frames

    # sort the files alphabetically assuming this will do them in the correct order
    files.sort()

    # create the videowriter - will create an .mp4
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

    # load and write the frames to the video
    for filename in tqdm(files, desc="Generating Video {}".format(video_path)):
        image = cv2.imread(filename)  # load the frame
        video.write(image)  # write the frame to the video

    video.release()  # release the video

    return video_path
