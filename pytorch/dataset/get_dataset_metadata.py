import json
import sys
from glob import glob


def parse(obj: dict):
    flattened = dict()
    for key in obj:
        if type(obj[key]) != dict:
            flattened[key] = obj[key]
        else:
            parsed = parse(obj[key])
            for k in parsed:
                flattened[f'{key}_{k}'] = parsed[k]
    return flattened


if __name__ == "__main__":
    opath = sys.argv[1]
    match_path = glob("timelines/*.json")[0]

    frames = []
    stats = set()
    with open(match_path, 'r') as f:
        s = f.read()
    match = json.loads(s)
    frame = match['info']['frames'][0]

    participant = frame['participantFrames']["1"]
    features = parse(participant)
   
    with open(opath, 'w') as f:
        for k in features:
            print(k)
            f.write(k + "\n")

