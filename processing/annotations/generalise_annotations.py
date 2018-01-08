"""
Converts annotation files (saves new copy) from informative (Players names and forehand/backhand) to spatial (near/far, right/left)
"""

import argparse
import json
import os

from config import config


def build_points_array(database):
    # build logical database of the form:
    # [point1, point2, ..., pointN] as:
    # [[set_score,game_score,point_score,start,end], [set_score,game_score,point_score,start,end], .., []]
    points = []
    for point in database['classes']['Point']:
        point_score = str(point['custom']['Score'])
        point_start = int(point['start'])
        point_end = int(point['end'])

        set_score = ''
        for set in database['classes']['Set']:
            score = str(set['custom']['Score'])
            set_start = int(set['start'])
            set_end = int(set['end'])
            if set_start <= point_start <= set_end or set_start <= point_end <= set_end:
                set_score = score

        game_score = ''
        for game in database['classes']['Game']:
            score = str(game['custom']['Score'])
            game_start = int(game['start'])
            game_end = int(game['end'])
            if game_start <= point_start <= game_end or game_start <= point_end <= game_end:
                game_score = score

        points.append([set_score, game_score, point_score, point_start, point_end, point['name']])
    return points


def convert(annotation_dir, files=None):
    error = False

    if files is None:
        files = os.listdir(annotation_dir)
        tmp = []
        for file in files:
            if file.find(".json") >= 0:
                tmp.append(file)
        files = tmp

    for file in files:
        path = os.path.join(annotation_dir, file)
        if os.path.isfile(path):
            print('File '+path+' exists. Loading it.')
            with open(path, 'r') as f:
                database = json.load(f)
        else:
            print('File '+path+' does not exist. Will make new database.')
            return None

        max_sets = 5
        hander = config.annotator.hander

        sets = {}
        for set in database['classes']['Set']:
            sets[set["start"]] = set['custom']['Near']

        sets_k = list(sets.keys())
        sets_k.sort()
        near = []
        for k in sets_k:
            near.append(sets[k])

        # build arrays
        points = build_points_array(database)

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

            # these need to be fixed!!!
            if found is None:
                error = True
                print('Error in annotation file: '+file+'\nEnsure all serve and hit events have some overlap with a point event.')
                print(serve)


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
            if sum(set_split) < max_sets:
                if sum(game_split) == 13: # Tiebreak
                    if len(point_split) > 1: # is a number and not 'Game'
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

            # these need to be fixed!!!
            if found is None:
                error = True
                print('Error in annotation file: '+file+'\nEnsure all serve and hit events have some overlap with a point event.')
                print(found)


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
            if sum(set_split) < max_sets:
                if sum(game_split) == 13:  # Tiebreak
                    if len(point_split) > 1:  # is a number and not 'Game'
                        point_split = [int(item) for item in point_split]
                        swaps += int((sum(point_split) - 1) / 6)

            if swaps % 2 == 0:  # positions same as start of set
                if hit['custom']['Player'] == near[sum(set_split) - 1]:
                    if hit['custom']['Player'] in hander[0]:  # right hander near
                        if hit['custom']['FB'] == 'Forehand':
                            hit['custom']['FB'] = 'Right'
                        elif hit['custom']['FB'] == 'Backhand':
                            hit['custom']['FB'] = 'Left'
                    elif hit['custom']['Player'] in hander[1]:  # left hander near
                        if hit['custom']['FB'] == 'Forehand':
                            hit['custom']['FB'] = 'Left'
                        elif hit['custom']['FB'] == 'Backhand':
                            hit['custom']['FB'] = 'Right'
                    hit['custom']['Player'] = 'Near'
                else:
                    if hit['custom']['Player'] in hander[0]:  # right hander far
                        if hit['custom']['FB'] == 'Forehand':
                            hit['custom']['FB'] = 'Left'
                        elif hit['custom']['FB'] == 'Backhand':
                            hit['custom']['FB'] = 'Right'
                    elif hit['custom']['Player'] in hander[1]:  # left hander far
                        if hit['custom']['FB'] == 'Forehand':
                            hit['custom']['FB'] = 'Right'
                        elif hit['custom']['FB'] == 'Backhand':
                            hit['custom']['FB'] = 'Left'
                    hit['custom']['Player'] = 'Far'
            else:  # positions opposite
                if hit['custom']['Player'] == near[sum(set_split) - 1]:
                    if hit['custom']['Player'] in hander[0]:  # right hander far
                        if hit['custom']['FB'] == 'Forehand':
                            hit['custom']['FB'] = 'Left'
                        elif hit['custom']['FB'] == 'Backhand':
                            hit['custom']['FB'] = 'Right'
                    elif hit['custom']['Player'] in hander[1]:  # left hander far
                        if hit['custom']['FB'] == 'Forehand':
                            hit['custom']['FB'] = 'Right'
                        elif hit['custom']['FB'] == 'Backhand':
                            hit['custom']['FB'] = 'Left'
                    hit['custom']['Player'] = 'Far'
                else:
                    if hit['custom']['Player'] in hander[0]:  # right hander near
                        if hit['custom']['FB'] == 'Forehand':
                            hit['custom']['FB'] = 'Right'
                        elif hit['custom']['FB'] == 'Backhand':
                            hit['custom']['FB'] = 'Left'
                    elif hit['custom']['Player'] in hander[1]:  # left hander near
                        if hit['custom']['FB'] == 'Forehand':
                            hit['custom']['FB'] = 'Left'
                        elif hit['custom']['FB'] == 'Backhand':
                            hit['custom']['FB'] = 'Right'
                    hit['custom']['Player'] = 'Near'

        if not error:

            path = os.path.join(annotation_dir, 'generalised', file)
            if not os.path.exists(os.path.join(annotation_dir, 'generalised')):
                os.makedirs(os.path.join(annotation_dir, 'generalised'))
            with open(path, 'w') as f:
                json.dump(database, f)

            print('File saved: ' + path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation_dir", type=str, default=config.directories.annotations,
                        help='Path to annotation directory')
    args = parser.parse_args()

    convert(annotation_dir=args.annotation_dir)
