from dataset.parse_dataset import match_2_timeframe
from models.timeframe import TimeframeModel
import torch

import json
import sys
import pandas as pd
from glob import glob
import os

import requests
import numpy as np
import matplotlib.pyplot as plt


def print_event(event, items):
    if event['type'] == 'BUILDING_KILL':
        s = ("Blue" if event["teamId"] < 150 else "Red") + " team "
        s += f'destroyed {event["buildingType"]} at {event["laneType"]}'
    if event['type'] == "ELITE_MONSTER_KILL":
        s = ("Blue" if event["killerTeamId"] < 150 else "Red") + " team "
        s += f'player {event["killerId"]} slayed {event["monsterType"]}'
    if event['type'] == "CHAMPION_KILL":
        s = f'Player {event["killerId"]} killed {event["victimId"]} with bounty {event["bounty"]}: streak {event["killStreakLength"]}'
    if event['type'] == "CHAMPION_SPECIAL_KILL":
        if event["killType"] == "KILL_MULTI":
            killType = {
                2: "Double",
                3: "Triple",
                4: "Quadra",
                5: "Penta"
            }.get(event["multiKillLength"])
            s = f'{killType} kill by Player {event["killerId"]}'
        else:
            s = f'{event["killType"]} by Player {event["killerId"]}'
    if event['type'] == 'ITEM_PURCHASED':
        s = f'Player {event["participantId"]} purchased item {items.get(str(event["itemId"]))["name"]}'
    print(s)

def download_items():
    resp = requests.get("https://ddragon.leagueoflegends.com/cdn/12.6.1/data/en_US/item.json")
    items = resp.json()
    with open('./items.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(items['data'], indent=4))


if __name__ == '__main__':
    with open("./dataset/stats_inter_list.txt", 'r') as f:
        stats = f.read().split('\n')

    if not os.path.exists('./items.json'):
        download_items()

    with open("./items.json", 'r') as f:
        items = json.loads(f.read())
    model_path, match_path, match_idx, include_item = sys.argv[1:]
    match_path = glob(os.path.join(match_path, '*.json'))[eval(match_idx)]
    model = TimeframeModel(89)
    model.load_state_dict(torch.load(model_path))
    print("Model created")
    model.eval()
    with open(match_path) as f:
        match = json.loads(f.read())
    
    frames = match['info']['frames']
    critical_events = ['CHAMPION_SPECIAL_KILL', 'CHAMPION_KILL', 'BUILDING_KILL', 'ELITE_MONSTER_KILL']
    if eval(include_item):
        critical_events.append('ITEM_PURCHASED')

    events = []
    for f in frames:
        fevents = []
        for e in f['events']:
            if e['type'] in critical_events:
                fevents.append(e)
        events.append(fevents)
    frames = match_2_timeframe(match, stats)
   
    cols = frames[0].keys()
    df = pd.DataFrame([[frame[c] for c in cols] for frame in frames], columns=cols)

    labels = df['label'].to_numpy('float32')
    df = df.drop('label', axis=1)
    frames = df.to_numpy(dtype='float32')

    metrics = []
    opp_metrics = []
    prev = 0.5
    with torch.no_grad():
        for i, x in enumerate(frames):
            output = model(torch.Tensor(x[np.newaxis, :]))[0].numpy()
            pred = output[0]
            if abs(prev - pred) > 0.15:
                print(f"\n{'='*5}Critical moment at minute {i + 1}")
                for e in events[i]:
                    print_event(e, items)
            prev = pred
            metrics.append(pred)
            opp_metrics.append(1 - pred)
    
    plt.plot(range(1, len(metrics) + 1), metrics, 'b', label='Blue Team')
    plt.plot(range(1, len(metrics) + 1), opp_metrics, 'r', label='Red Team')
    plt.xlabel("Minutes after game start")
    plt.ylabel("Win Probability")
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f'Game that {"Red" if labels[0] == 0 else "Blue"} team won')
    plt.show()
