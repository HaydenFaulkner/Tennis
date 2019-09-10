"""
Pre-processing steps for the annotations data, expects un-generalised .jsons straight from the annotator

Does the following:
1. Generates slice .txt files for each .json annotation file
2. Generalises the .json annotation files from player names and forehand/backhand to near/far and left/right
3. Generates label .txt files for each generalised .json annotation file

"""


from absl import app, flags, logging
from absl.flags import FLAGS
import json
import os
from tqdm import tqdm

MAX_SETS = 5
HANDER = [['Federer', 'Williams', 'Sharapova', 'Djokovic', 'Tsonga', 'Zvonareva', 'Del Potro', 'Azarenka'],
          ['Nadal']]  # [RIGHT,LEFT] handedness of players


def generate_slices(annotations_dir, slices_dir, videos):
    """
    Generate slice .txt file in the slices_dir from the .json annotation file provided

    Args:
        annotations_dir (str): path to the directory where the raw .json files are
        slices_dir (str): the path to the directory to save the slices .txt files
        videos (list): list of json filenames to do, without their extension

    Returns:
        int: 1 if success, or None if error occurred
    """

    annotations_dir = os.path.normpath(annotations_dir)  # make the paths OS (Windows) compatible

    for video in tqdm(videos, desc='Generating slices'):
        annotation_path = os.path.join(annotations_dir, video + '.json')

        # check annotation file exists, throw error if not
        if not os.path.exists(annotation_path):
            logging.error("File {} does not exist.".format(annotation_path))
            return None

        # loads the annotation .json file
        with open(annotation_path, 'r') as f:
            database = json.load(f)

        # ensure the database contains the needed information
        if "classes" not in database.keys() or "USE" not in database["classes"].keys():
            logging.error("Database {} is not structured correctly, can't extract.\n"
                          "Needs 'classes' and 'USE' categories.".format(annotation_path))
            return None

        # make the slices directory if it doesn't exist
        os.makedirs(slices_dir, exist_ok=True)

        # save out the slices .txt file
        save_path = os.path.join(slices_dir, video + ".txt")
        with open(save_path, 'w') as f:
            for slice_ in database['classes']['USE']:
                f.write("{} {}\n".format(int(slice_['start']), int(slice_['end'])))

    return 1


def generate_points_list(database):
    """
    Generates a list of points from the database dictionary

    Args:
        database (dict): a database dict that was loaded from a .json annotation file

    Returns:
        list: of lists being points as [[set_score, game_score, point_score, start, end], ..., ]
    """

    points = []
    for point in database['classes']['Point']:
        point_score = str(point['custom']['Score'])  # the point score after point as 30-0 or 40-A or Game or Deuce etc.
        point_start = int(point['start'])  # the frame the point starts
        point_end = int(point['end'])  # the frame the point ends

        # make the set score by working out where a point fits in time eg. 1-0 or 0-2 etc.
        set_score = ''
        for set_ in database['classes']['Set']:
            score = str(set_['custom']['Score'])
            set_start = int(set_['start'])
            set_end = int(set_['end'])
            if set_start <= point_start <= set_end or set_start <= point_end <= set_end:
                set_score = score

        # similarily make the game score using point frame stamp eg. 1-0 or 4-4 or 6-4 etc.
        game_score = ''
        for game in database['classes']['Game']:
            score = str(game['custom']['Score'])
            game_start = int(game['start'])
            game_end = int(game['end'])
            if game_start <= point_start <= game_end or game_start <= point_end <= game_end:
                game_score = score

        # add to the points list
        points.append([set_score, game_score, point_score, point_start, point_end, point['name']])

    return points


def generalise_jsons(annotations_dir, generalised_dir, videos):
    """
    Generalises the .json annotation files from player names and forehand/backhand to near/far and left/right

    Args:
        annotations_dir (str): path to the directory where the raw .json files are
        generalised_dir (str): path to the generalised output directory
        videos (list): list of json filenames to do, without their extension

    Returns:
        int: 1 if success, or None if error occurred
    """

    annotation_path = os.path.normpath(annotations_dir)  # make the paths OS (Windows) compatible

    # check annotation file exists, throw error if not
    if not os.path.exists(annotation_path):
        logging.error("File {} does not exist".format(annotation_path))
        return None

    for video in tqdm(videos, desc='Generalising jsons'):
        annotation_path = os.path.join(annotations_dir, video + '.json')
        # check annotation file exists, throw error if not
        if not os.path.exists(annotation_path):
            logging.error("File {} does not exist".format(annotation_path))
            return None

        # load the annotation .json file
        with open(annotation_path, 'r') as f:
            database = json.load(f)

        # make a sets dictionary containing the name of the closest player at start of set keyed by frame number
        sets = {}
        for set_ in database['classes']['Set']:
            sets[set_["start"]] = set_['custom']['Near']

        # a list of the near players at the start of each set
        near = [sets[key] for key in sorted(sets.keys())]

        # build arrays
        points = generate_points_list(database)

        players = set()
        for game in database['classes']['Game']:
            players.add(game['custom']['Winner'])
        players = list(players)
        assert len(players) == 2

        # fix serves
        for serve in database['classes']['Serve']:
            start = int(serve['start'])
            end = int(serve['end'])
            middle = start + (end - start) / 2

            # check middle part of event is within either a point, a fault or a let
            found = None
            closest = 100000000
            for point in points:
                if point[3] <= middle <= point[4]:
                    found = point
                    break
                else:
                    if point[3] - end < closest:
                        found = point

            # couldn't fine which point this serve belongs to!! oh no! these need to be fixed!!!
            if found is None:
                logging.error("Error in annotation file: {}\nEnsure all serve and hit events have some overlap with a "
                              "point event.\nCaused by: {}".format(annotation_path, serve))
                return None

            set_split = found[0].split('-')
            set_split = [int(item) for item in set_split]
            game_split = found[1].split('-')
            game_split = [int(item) for item in game_split]
            point_split = found[2].split('-')

            # how many end changes based on score
            swaps = 0
            # this is the number of swaps so far in set
            swaps += int(sum(game_split) / 2)

            # add any swaps in tiebreak situation
            if sum(set_split) < MAX_SETS:
                if sum(game_split) == 13:  # Tiebreak
                    if len(point_split) > 1:  # is a number and not 'Game'
                        point_split = [int(item) for item in point_split]
                        swaps += int((sum(point_split) - 1) / 6)

            if swaps % 2 == 0:  # positions same as start of set
                assert serve['custom']['Player'] in players, '{} not a valid player name {}' \
                                                             ''.format(serve['custom']['Player'], near)
                if serve['custom']['Player'] == near[sum(set_split) - 1]:
                    serve['custom']['Player'] = 'Near'
                else:
                    serve['custom']['Player'] = 'Far'
            else:  # positions opposite
                if serve['custom']['Player'] == near[sum(set_split) - 1]:
                    serve['custom']['Player'] = 'Far'
                else:
                    serve['custom']['Player'] = 'Near'

        # fix hits
        for hit in database['classes']['Hit']:
            start = int(hit['start'])
            end = int(hit['end'])
            middle = start + (end - start) / 2

            # check middle part of event is within either a point, a fault or a let
            found = None
            closest = 100000000
            for point in points:
                if point[3] <= middle <= point[4]:
                    found = point
                    break
                else:
                    if point[3] - end < closest:
                        found = point

            # couldn't fine which point this hit belongs to!! oh no! these need to be fixed!!!
            if found is None:
                logging.error("Error in annotation file: {}\nEnsure all serve and hit events have some overlap with a "
                              "point event.\nCaused by: {}".format(annotation_path, hit))
                return None

            set_split = found[0].split('-')
            set_split = [int(item) for item in set_split]
            game_split = found[1].split('-')
            game_split = [int(item) for item in game_split]
            point_split = found[2].split('-')

            # how many end changes based on score
            swaps = 0
            # this is the number of swaps so far in set
            swaps += int(sum(game_split) / 2)

            # add any swaps in tiebreak situation
            if sum(set_split) < MAX_SETS:
                if sum(game_split) == 13:  # Tiebreak
                    if len(point_split) > 1:  # is a number and not 'Game'
                        point_split = [int(item) for item in point_split]
                        swaps += int((sum(point_split) - 1) / 6)

            if swaps % 2 == 0:  # positions same as start of set
                assert hit['custom']['Player'] in players, '{} not a valid player name {}' \
                                                           ''.format(hit['custom']['Player'], near)
                if hit['custom']['Player'] == near[sum(set_split) - 1]:
                    if hit['custom']['Player'] in HANDER[0]:  # right HANDER near
                        if hit['custom']['Side'] == 'Forehand':
                            hit['custom']['Side'] = 'Right'
                        elif hit['custom']['Side'] == 'Backhand':
                            hit['custom']['Side'] = 'Left'
                    elif hit['custom']['Player'] in HANDER[1]:  # left HANDER near
                        if hit['custom']['Side'] == 'Forehand':
                            hit['custom']['Side'] = 'Left'
                        elif hit['custom']['Side'] == 'Backhand':
                            hit['custom']['Side'] = 'Right'
                    hit['custom']['Player'] = 'Near'
                else:
                    if hit['custom']['Player'] in HANDER[0]:  # right HANDER far
                        if hit['custom']['Side'] == 'Forehand':
                            hit['custom']['Side'] = 'Left'
                        elif hit['custom']['Side'] == 'Backhand':
                            hit['custom']['Side'] = 'Right'
                    elif hit['custom']['Player'] in HANDER[1]:  # left HANDER far
                        if hit['custom']['Side'] == 'Forehand':
                            hit['custom']['Side'] = 'Right'
                        elif hit['custom']['Side'] == 'Backhand':
                            hit['custom']['Side'] = 'Left'
                    hit['custom']['Player'] = 'Far'

            else:  # positions opposite
                if hit['custom']['Player'] == near[sum(set_split) - 1]:
                    if hit['custom']['Player'] in HANDER[0]:  # right HANDER far
                        if hit['custom']['Side'] == 'Forehand':
                            hit['custom']['Side'] = 'Left'
                        elif hit['custom']['Side'] == 'Backhand':
                            hit['custom']['Side'] = 'Right'
                    elif hit['custom']['Player'] in HANDER[1]:  # left HANDER far
                        if hit['custom']['Side'] == 'Forehand':
                            hit['custom']['Side'] = 'Right'
                        elif hit['custom']['Side'] == 'Backhand':
                            hit['custom']['Side'] = 'Left'
                    hit['custom']['Player'] = 'Far'
                else:
                    if hit['custom']['Player'] in HANDER[0]:  # right HANDER near
                        if hit['custom']['Side'] == 'Forehand':
                            hit['custom']['Side'] = 'Right'
                        elif hit['custom']['Side'] == 'Backhand':
                            hit['custom']['Side'] = 'Left'
                    elif hit['custom']['Player'] in HANDER[1]:  # left HANDER near
                        if hit['custom']['Side'] == 'Forehand':
                            hit['custom']['Side'] = 'Left'
                        elif hit['custom']['Side'] == 'Backhand':
                            hit['custom']['Side'] = 'Right'
                    hit['custom']['Player'] = 'Near'

        # write out the generalised version of the annotation file
        path = os.path.join(generalised_dir, video + '.json')
        os.makedirs(generalised_dir, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(database, f)

    return 1


def generate_labels(generalised_dir, labels_dir, videos):
    """
    generate labels .txt files from .json generalised annotation files
    Args:
        generalised_dir (str): path to the annotations directory, where the generalised annotation .json files are
        labels_dir (str): path to the labels directory, where the .txt files will be saved
        videos (list): list of json filenames to do, without their extension

    Returns:
        None
    """

    # load in the class names
    assert os.path.exists(os.path.join('data', 'classes.names'))
    with open(os.path.join('data', 'classes.names'), 'r') as f:
        lines = f.readlines()
    classes = [line.rstrip() for line in lines]

    annotations_dir = os.path.normpath(generalised_dir)  # make the paths OS (Windows) compatible

    for video in tqdm(videos, desc='Generating labels'):
        labels = dict()
        frames = dict()
        for class_ in classes:
            frames[class_] = []

        annotation_path = os.path.join(annotations_dir, video + '.json')
        # check annotation file exists, throw error if not
        if not os.path.exists(annotation_path):
            logging.error("File {} does not exist".format(annotation_path))
            return None

        # load the annotation .json file
        with open(annotation_path, 'r') as f:
            database = json.load(f)

        # build the frames dict containing the frame numbers for each class
        for hit in database['classes']['Hit']:
            if hit['custom']['Player'] == 'Far':
                if hit['custom']['Side'] == 'Right':
                    frames['HFR'] += list(range(int(hit['start']), int(hit['end'])))
                elif hit['custom']['Side'] == 'Left':
                    frames['HFL'] += list(range(int(hit['start']), int(hit['end'])))
                else:
                    return AttributeError

            elif hit['custom']['Player'] == 'Near':
                if hit['custom']['Side'] == 'Right':
                    frames['HNR'] += list(range(int(hit['start']), int(hit['end'])))
                elif hit['custom']['Side'] == 'Left':
                    frames['HNL'] += list(range(int(hit['start']), int(hit['end'])))
                else:
                    return AttributeError
            else:
                return AttributeError

        for serve in database['classes']['Serve']:
            if serve['custom']['Player'] == 'Far':
                if serve['custom']['Result'] == 'In':
                    frames['SFI'] += list(range(int(serve['start']), int(serve['end'])))
                elif serve['custom']['Result'] == 'Fault':
                    frames['SFF'] += list(range(int(serve['start']), int(serve['end'])))
                elif serve['custom']['Result'] == 'Let':
                    frames['SFL'] += list(range(int(serve['start']), int(serve['end'])))
                else:
                    return AttributeError

            elif serve['custom']['Player'] == 'Near':
                if serve['custom']['Result'] == 'In':
                    frames['SNI'] += list(range(int(serve['start']), int(serve['end'])))
                elif serve['custom']['Result'] == 'Fault':
                    frames['SNF'] += list(range(int(serve['start']), int(serve['end'])))
                elif serve['custom']['Result'] == 'Let':
                    frames['SNL'] += list(range(int(serve['start']), int(serve['end'])))
                else:
                    return AttributeError
            else:
                return AttributeError

        # populate the labels dict with the classes
        start = database['classes']['USE'][0]['start']
        end = database['classes']['USE'][0]['end']

        for frame in range(start, end):
            labels[frame] = 'OTH'
            for k, v in frames.items():
                if frame in v:
                    labels[frame] = k

        # make the labels directory if it doesn't exist
        os.makedirs(labels_dir, exist_ok=True)

        # save out the labels files
        with open(os.path.join(labels_dir, video + '.txt'), 'w') as f:
            for k, v in sorted(labels.items()):
                f.write("{}\t{}\n".format(k, v))


def main(_argv):

    generate_slices(annotations_dir=FLAGS.annotations_dir, slices_dir=FLAGS.slices_dir, videos=FLAGS.videos)
    generalise_jsons(annotations_dir=FLAGS.annotations_dir, generalised_dir=FLAGS.generalised_dir, videos=FLAGS.videos)
    generate_labels(generalised_dir=FLAGS.generalised_dir, labels_dir=FLAGS.labels_dir, videos=FLAGS.videos)


if __name__ == "__main__":

    flags.DEFINE_string('annotations_dir', 'data/annotations',
                        'Path to the annotations directory, where the raw annotation .json files are')
    flags.DEFINE_string('generalised_dir', 'data/annotations/generalised',
                        'Path to the annotations directory, where the generalised annotation .json files are')
    flags.DEFINE_string('slices_dir', 'data/annotations/slices',
                        'Path to the slices directory, where the .txt files will be saved')
    flags.DEFINE_string('labels_dir', 'data/annotations/labels',
                        'Path to the labels directory, where the .txt files will be saved')
    flags.DEFINE_list('videos', 'V006,V007,V008,V009,V010',
                      'List of json files to do, without their extension')

    try:
        app.run(main)
    except SystemExit:
        pass
