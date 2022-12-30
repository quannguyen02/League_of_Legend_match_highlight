import json
from glob import glob
import sys

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def match_2_timeframe(match, stats):
    scaler = MinMaxScaler()
    frames = match['info']['frames']
    timeframes = []

    # Find Label
    events = frames[-1]['events']
    winning_team = 100
    for e in events:
        if e['type'] == 'GAME_END':
            winning_team = e['winningTeam']
            break

    for i, frame in enumerate(frames):
        x = {'label': 1 if winning_team < 150 else 0, 'game_percentage': round(i / (len(frames)-1), 2)}
        participants = frame['participantFrames']
        for s in stats:
            attrs = s.split('_')
            team1_avg = 0
            team2_avg = 0
            if len(attrs) == 1:  
                for i, p in participants.items():
                    if int(i) < 6:
                        team1_avg += p[s]
                    else:
                        team2_avg += p[s]
            else:
                a1, a2 = attrs
                for i, p in participants.items():
                    if int(i) < 6:
                        team1_avg += p[a1][a2]
                    else:
                        team2_avg += p[a1][a2]
            team1_avg /= 5
            team2_avg /= 5
            x[f'team1_avg_{s}'] = team1_avg
            x[f'team2_avg_{s}'] = team2_avg
        timeframes.append(x)
    for s in stats:
        data = [[tf[f'team1_avg_{s}']] for tf in timeframes] + [[tf[f'team2_avg_{s}']] for tf in timeframes]
        scaler.fit(data)
        data = scaler.transform(data)
        for i, tf in enumerate(timeframes):
            tf[f'team1_avg_{s}'] = data[i][0]
            tf[f'team2_avg_{s}'] = data[i + len(timeframes)][0]
    return timeframes


def save_frames_to_csv(frames, opath):
    cols = frames[0].keys()
    df = pd.DataFrame([[frame[c] for c in cols] for frame in frames], columns=cols)
    df.to_csv(opath, index=False)


if __name__ == "__main__":
    ipath, opath, attr_file = sys.argv[1:]
    matches = glob(f"{ipath}/*.json")

    with open(attr_file, 'r') as f:
        stats = f.read().split('\n')
    frames = []

    for match_path in tqdm(matches):
        with open(match_path, 'r') as f:
            s = f.read()
        match = json.loads(s)
        try:
            frames += match_2_timeframe(match, stats)
        except Exception as e:
            print(match_path)
    save_frames_to_csv(frames, opath)
