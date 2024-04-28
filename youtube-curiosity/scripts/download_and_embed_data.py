import csv
import json
import random

import sys
import os

import pickle
from tqdm import tqdm

sys.path.append("..")

from dataset.download import prepare_video_no_clip
from dataset.tokenization.tokenize_image import tokenize_image

def get_sample_videos(data_dir_path):
    # read and sample video IDs
    with open("../dataset/youtube_videos.csv") as f:
        reader = csv.reader(f)
        video_ids = [row[0] for row in reader]

        sample_videos = random.sample(video_ids, 840)        

    # write samples to file
    with open(f"{data_dir_path}train_videos.json", "w") as f:
        videos = {
            "train": sample_videos[:720],
            "train_small": sample_videos[720:]
        }

        json.dump(videos, f)

        return videos
    
def get_test_videos(data_dir_path):
    # read and sample video IDs
    with open("../dataset/youtube_videos.csv") as f:
        reader = csv.reader(f)
        video_ids = [row[0] for row in reader]

        sample_videos = random.sample(video_ids, 20)

    # write samples to file
    with open(f"{data_dir_path}test_videos.json", "w") as f:
        json.dump(sample_videos, f)

        return sample_videos

def collect_data(video_ids, filepath, data_dir_path):
    acc = []

    with open(filepath, "w") as f:
        for video_id in tqdm(video_ids, desc="Processing videos"):
            n_frames = prepare_video_no_clip(video_id, data_dir_path)
            
            prev_tokenized = None

            for frame_number in range(n_frames):
                tokenized = tokenize_image(video_id, frame_number, data_dir_path)

                if prev_tokenized is not None and tokenized is not None:
                    # add to training set
                    acc.append({
                        'video_id': video_id,
                        'prev_frame': json.dumps(prev_tokenized.tolist()),
                        'frame': json.dumps(tokenized.tolist()),
                    })

                if tokenized is not None:
                    prev_tokenized = tokenized

        json.dump(acc, f)

def collate_data(n_segments, data_dir_path):
    data = []

    for i in range(n_segments):
        with open(f"{data_dir_path}dataset/train-{i}.json") as f:
            data.extend(json.load(f))

    with open(f"{data}dataset/train.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    video_ids = get_sample_videos()

    with open(f"/grogu/user/mhzhou/youtube-curiosity/train_videos.json") as f:
        video_ids = json.load(f)

        n_segments = 10

        for i in range(n_segments):
            n = len(video_ids["train"]) // n_segments

            collect_data(
                video_ids["train"][i * n : (i + 1) * n],
                f"/grogu/user/mhzhou/youtube-curiosity/dataset/train-{i}.json",
                data_dir_path="/grogu/user/mhzhou/youtube-curiosity/",
            )

        collect_data(
            video_ids["train_small"],
            "/grogu/user/mhzhou/youtube-curiosity/dataset/train-small.json",
            data_dir_path="/grogu/user/mhzzhou/youtube-curiosity/",
        )

        collate_data(n_segments, data_dir_path="/grogu/user/mhzhou/youtube-curiosity/")

    video_ids = get_test_videos()

    collect_data(video_ids, "/grogu/user/mhzhou/youtube-curiosity/dataset/test.json", data_dir_path="/grogu/user/mhzhou/youtube-curiosity/")
