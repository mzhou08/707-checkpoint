import csv
import pytube
import json
import time
import tqdm
import os

import multiprocessing

def video_id_to_channel(video_id):
    """
    returns the channel ID for a given video ID
    """
    yt = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
    while True:
        try:
            return yt.channel_id
        except:
            print(f"Failed to get channel for video {video_id}")

def get_howto100m_video_ids(howto100m_caption_path):
    """
    Returns a list of video ids from the HowTo100M dataset
    """
    ht100m = json.load(open(howto100m_caption_path))
    return list(ht100m.keys())

def get_hdvila100m_video_ids(hdvila_folder_path):
    """
    Returns a list of video ids from the HD-VILA-100M dataset
    """
    video_ids = []

    for file in os.listdir(hdvila_folder_path):
        if file.endswith(".jsonl"):
            with open(os.path.join(hdvila_folder_path, file)) as f:
                for line in tqdm.tqdm(f, desc=f"Reading {file}"):
                    video = json.loads(line.strip())
                    video_ids.append(video["video_id"])

    return video_ids

def get_yttemporal1b_video_ids(yttemporal1b_folder_path):
    """
    Returns a list of video ids from the YT-Temporal-1B dataset
    https://rowanzellers.com/merlotreserve/
    """
    video_ids = []

    with open(os.path.join(yttemporal1b_folder_path, "yttemporal1b_ids_train.csv")) as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            video_ids.append(row[0])

    with open(os.path.join(yttemporal1b_folder_path, "yttemporal1b_ids_val.csv")) as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            video_ids.append(row[0])

    return video_ids

def get_vlog_ids(vlog_folder_path):
    """
    Returns a list of video ids from the VLOG dataset
    https://web.eecs.umich.edu/~fouhey//2017/VLOG/
    """
    video_ids = []

    with open(os.path.join(vlog_folder_path, "meta/youtube_links.txt")) as f:
        for line in f:
            youtube_link = line.strip().split(" ")[0]
            video_id = youtube_link.split("https://www.youtube.com/watch?v=")[1]
            video_ids.append(video_id)
    
    return video_ids

def get_internvid_ids(internvid_folder_path):
    """
    Returns a list of video ids from the InternVid dataset
    https://arxiv.org/abs/2307.06942
    """

    video_ids = set()

    with open(os.path.join(internvid_folder_path, "InternVid-18M-aes.jsonl")) as f:
        for line in tqdm.tqdm(f):
            j = json.loads(line.strip())
            video_ids.add(j["YoutubeID"])

    with open(os.path.join(internvid_folder_path, "InternVid-10M-flt.jsonl")) as f:
        for line in tqdm.tqdm(f):
            j = json.loads(line.strip())
            video_ids.add(j["YoutubeID"])

    with open(os.path.join(internvid_folder_path, "InternVid-10M-DIV.jsonl")) as f:
        for line in tqdm.tqdm(f):
            j = json.loads(line.strip())
            video_ids.add(j["YoutubeID"])

    return list(video_ids)

def get_channel_ids(video_ids, output_file, num_segments=10):
    """
    Given a list of video IDs, retrieves the corresponding channel IDs and writes them to a json file

    Args:
    - video_ids: a list of video ids
    - output_file: file to write channel IDs to
    """

    # 10 segments per dataset
    for s in range(num_segments):

        start = time.time()
        channels = set()


        video_ids_segment = video_ids[
            s * len(video_ids) // num_segments
            : (s + 1) * len(video_ids) // num_segments
        ]

        with multiprocessing.Pool() as pool:
            max_ = len(video_ids_segment)
            with tqdm.tqdm(total=max_) as pbar:
                for _, channel in enumerate(pool.imap_unordered(video_id_to_channel, video_ids_segment)):
                    channels.add(channel)
                    pbar.update()

        end = time.time()
        n_channels = len(channels)

        print(f"retrieved {n_channels} channels in {end - start} seconds. {(end - start) / n_channels} seconds per channel")

        # write channel ids to json
        output_file_path = f"{output_file}_{s}.json"

        json.dump(
            [channel_id for channel_id in channels if channel_id is not None],
            open(output_file_path, "w"),
        )


if __name__ == "__main__":
    BASE_FILE_PATH = "/grogu/user/mhzhou/youtube-curiosity/dataset/scraping/"

    # ht100m_video_ids = get_howto100m_video_ids("/grogu/user/mhzhou/playground/howto100m/caption.json")
    # get_channel_ids(ht100m_video_ids, f"{BASE_FILE_PATH}/howto100m_channels")

    # wordnet_video_ids = json.load(open("/grogu/user/mhzhou/youtube-curiosity/dataset/scraping/wordnet_videos.json"))
    # get_channel_ids(wordnet_video_ids, f"{BASE_FILE_PATH}/wordnet_channels")

    # vlog_videos_ids = get_vlog_ids("/grogu/user/mhzhou/playground/vlog")
    # get_channel_ids(vlog_videos_ids, f"{BASE_FILE_PATH}/vlog_channels)

    # hd_vila_video_ids = get_hdvila100m_video_ids("/grogu/user/mhzhou/playground/hdvila")
    # get_channel_ids(hd_vila_video_ids, f"{BASE_FILE_PATH}/hdvila100m_channels")


    # internvid_video_ids = get_internvid_ids("/grogu/user/mhzhou/playground/internvid10m")
    # get_channel_ids(internvid_video_ids, f"{BASE_FILE_PATH}/internvid_channels")
    
    # all above this point are done

    # yt_temporal1b_video_ids = get_yttemporal1b_video_ids("/grogu/user/mhzhou/playground/yttemporal1b")
    # get_channel_ids(yt_temporal1b_video_ids, f"{BASE_FILE_PATH}/yttemporal1b_channels")

    commoncrawl_video_ids = json.load(open("/grogu/user/mhzhou/youtube-curiosity/dataset/scraping/commoncrawl_videos.json"))[1:]
    get_channel_ids(commoncrawl_video_ids, f"{BASE_FILE_PATH}/commoncrawl_channels")
