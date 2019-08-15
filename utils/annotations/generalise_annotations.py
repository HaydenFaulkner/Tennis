"""
Converts annotation files (saves new copy) from informative (Players names and forehand/backhand) to spatial (near/far, right/left)
"""

from absl import app, flags, logging
from absl.flags import FLAGS
import json
import os

MAX_SETS = 5
HANDER = [['Federer', 'Williams', 'Sharapova', 'Djokovic', 'Tsonga', 'Zvonareva', 'Del Potro', 'Azarenka'],
          ['Nadal']]  # [RIGHT,LEFT]


def generate_points_list(database):
    """
    Generates a list of points from the database dictionary

    :param database: a database dict that was loaded from a .json annotation file
    :return: a list of points as [[set_score, game_score, point_score, start, end], ..., ]
    """

    points = []
    for point in database['classes']['Point']:
        point_score = str(point['custom']['Score'])  # the point score after point as 30-0 or 40-A or Game or Deuce etc.
        point_start = int(point['start'])  # the frame the point starts
        point_end = int(point['end'])  # the frame the point ends

        # make the set score by working out where a point fits in time eg. 1-0 or 0-2 etc.
        set_score = ''
        for set in database['classes']['Set']:
            score = str(set['custom']['Score'])
            set_start = int(set['start'])
            set_end = int(set['end'])
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


def generalise_json(annotation_path):
    """
    Convert a .json annotation file into a generalised file without player names

    :param annotation_path: path to the .json file
    :return: path that the new generalised .json is saved, or None if error occurred
    """

    annotation_path = os.path.normpath(annotation_path)  # make the paths OS (Windows) compatible
    annotation_dir, annotation_filename = os.path.split(annotation_path)  # get the dir and filename
    
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

    # fix serves
    for serve in database['classes']['Serve']:
        start = int(serve['start'])
        end = int(serve['end'])
        middle = start + (end-start)/2

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
            logging.error("Error in annotation file: {}\nEnsure all serve and hit "
                          "events have some overlap with a point event.\nCaused by: {}".format(annotation_path, serve))
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
                    swaps += int((sum(point_split)-1) / 6)

        if swaps % 2 == 0:  # positions same as start of set
            if serve['custom']['Player'] == near[sum(set_split)-1]:
                serve['custom']['Player'] = 'Near'
            else:
                serve['custom']['Player'] = 'Far'
        else:  # positions opposite
            if serve['custom']['Player'] == near[sum(set_split)-1]:
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
            logging.error("Error in annotation file: {}\nEnsure all serve and hit "
                          "events have some overlap with a point event.\nCaused by: {}".format(annotation_path, hit))
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
            if hit['custom']['Player'] == near[sum(set_split) - 1]:
                if hit['custom']['Player'] in HANDER[0]:  # right HANDER near
                    if hit['custom']['FB'] == 'Forehand':
                        hit['custom']['FB'] = 'Right'
                    elif hit['custom']['FB'] == 'Backhand':
                        hit['custom']['FB'] = 'Left'
                elif hit['custom']['Player'] in HANDER[1]:  # left HANDER near
                    if hit['custom']['FB'] == 'Forehand':
                        hit['custom']['FB'] = 'Left'
                    elif hit['custom']['FB'] == 'Backhand':
                        hit['custom']['FB'] = 'Right'
                hit['custom']['Player'] = 'Near'
            else:
                if hit['custom']['Player'] in HANDER[0]:  # right HANDER far
                    if hit['custom']['FB'] == 'Forehand':
                        hit['custom']['FB'] = 'Left'
                    elif hit['custom']['FB'] == 'Backhand':
                        hit['custom']['FB'] = 'Right'
                elif hit['custom']['Player'] in HANDER[1]:  # left HANDER far
                    if hit['custom']['FB'] == 'Forehand':
                        hit['custom']['FB'] = 'Right'
                    elif hit['custom']['FB'] == 'Backhand':
                        hit['custom']['FB'] = 'Left'
                hit['custom']['Player'] = 'Far'
        else:  # positions opposite
            if hit['custom']['Player'] == near[sum(set_split) - 1]:
                if hit['custom']['Player'] in HANDER[0]:  # right HANDER far
                    if hit['custom']['FB'] == 'Forehand':
                        hit['custom']['FB'] = 'Left'
                    elif hit['custom']['FB'] == 'Backhand':
                        hit['custom']['FB'] = 'Right'
                elif hit['custom']['Player'] in HANDER[1]:  # left HANDER far
                    if hit['custom']['FB'] == 'Forehand':
                        hit['custom']['FB'] = 'Right'
                    elif hit['custom']['FB'] == 'Backhand':
                        hit['custom']['FB'] = 'Left'
                hit['custom']['Player'] = 'Far'
            else:
                if hit['custom']['Player'] in HANDER[0]:  # right HANDER near
                    if hit['custom']['FB'] == 'Forehand':
                        hit['custom']['FB'] = 'Right'
                    elif hit['custom']['FB'] == 'Backhand':
                        hit['custom']['FB'] = 'Left'
                elif hit['custom']['Player'] in HANDER[1]:  # left HANDER near
                    if hit['custom']['FB'] == 'Forehand':
                        hit['custom']['FB'] = 'Left'
                    elif hit['custom']['FB'] == 'Backhand':
                        hit['custom']['FB'] = 'Right'
                hit['custom']['Player'] = 'Near'

    # write out the generalised version of the annotation file
    path = os.path.join(annotation_dir, 'generalised', annotation_filename)
    os.makedirs(os.path.join(annotation_dir, 'generalised'), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(database, f)

    logging.info("Generalised annotation file successfully save to: {}".format(path))

    return path


def main(_argv):  # main function
    generalise_json(annotation_path=FLAGS.annotation_path)  # just run with the specified .json


if __name__ == "__main__":

    flags.DEFINE_string('annotation_path', 'data/annotations/V001.json',
                        'Path of the .json annotation file to generalise.')

    try:
        app.run(main)
    except SystemExit:
        pass

