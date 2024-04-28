import json
import multiprocessing
import pytube
import os
import time
import tqdm

def get_video_metadata(video_id):
    yt = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")

    if "videoDetails" not in yt.vid_info:
        print(f"Failed to get metadata for video {video_id}")
        return {}
    
    elif yt.vid_info["videoDetails"].get("isPrivate", False):
        print(f"Video {video_id} is private")
        return {}

    try:
        views = yt.views
    except TypeError:
        views = None

    res = {
        "video_id": video_id,
        "title": yt.title,
        "description": "" if yt.description is None else yt.description,
        "keywords": yt.keywords,
        "rating": yt.rating,
        "thumbnail_urls": [d["url"] for d in yt.vid_info["videoDetails"].get("thumbnail", {}).get("thumbnails", [])],
        "views": views,
        # "vid_info": yt.vid_info,
        "channel_id": yt.channel_id,    # may be None for shorts
        "author": yt.author,      # may be None for shorts
        "publish_date": yt.publish_date,
        "length": yt.length,
    }

    return res

def get_channel_video_data(channel_id):
    yt = pytube.Channel(f"https://www.youtube.com/channel/{channel_id}")
    video_urls = yt.video_urls

    video_data = []

    for video_url in video_urls:
        try:
            video_id = video_url.split("https://www.youtube.com/watch?v=")[-1]
            metadata = get_video_metadata(video_id)
            if metadata != {}:
                video_data.append(metadata)
        except Exception as e:
            print(f"error {e} on video {video_url} from channel {channel_id}")
            continue
    return channel_id, video_data

def get_channel_video_ids(channel_id):
    yt = pytube.Channel(f"https://www.youtube.com/channel/{channel_id}")
    video_urls = yt.video_urls

    video_ids = []

    for video_url in video_urls:
        video_id = video_url.split("https://www.youtube.com/watch?v=")[-1]
        video_ids.append(video_id)

    time.sleep(1)

    return channel_id, video_ids


def process_channels_file(channels_file, start=0, num_segments=20):
    """
    Args:
    - channels_file: path to a json file containing a list of channel ids

    For each channel ID in channels_file, gets all of its videos and writes
    their video metadata to /videos/{channels_file}_video_data.json
    """
    channels = list(json.load(open(channels_file)))

    base_name = os.path.basename(channels_file)
    filename = os.path.splitext(base_name)[0]
    base_folder = os.path.dirname(channels_file)

    with multiprocessing.Pool() as pool:

        for i in range(start, num_segments):
            channels_subset = channels[
                i * len(channels) // num_segments
                : (i + 1) * len(channels) // num_segments
            ]

            video_data = {}
            video_ids = []

            start = time.time()

            max_ = len(channels_subset)
            with tqdm.tqdm(total=max_, desc=f"processing {filename}, segment {i}") as pbar:
                # for i, res in enumerate(pool.imap_unordered(get_channel_video_data, channels_subset)):
                for res in pool.imap_unordered(get_channel_video_ids, channels_subset):
                    channel_id, channel_videos = res
                    video_data[channel_id] = channel_videos
                    video_ids += channel_videos
                    pbar.update()

            end = time.time()

            print(f"retrieved {len(video_ids)} videos from {len(channels_subset)} channels in {end - start} seconds. {(end - start) / len(channels_subset)} seconds per channel")

            # output_file = os.path.join(base_folder, f"videos/{filename}_video_data_{i}.json")
            video_ids_output_file = os.path.join(base_folder, f"videos/{filename}_video_ids_{i}.json")
            channel_to_video_output_file = os.path.join(base_folder, f"videos/{filename}_channel_to_video_{i}.json")

            json.dump(
                video_data,
                open(channel_to_video_output_file, "w"),
            )

            json.dump(
                video_ids,
                open(video_ids_output_file, "w"),
            )

            # write channel ids to json
            # json.dump(
            #     video_data,
            #     open(output_file, "w"),
            # )
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--channels_file", type=str, required=True)
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_segments", type=int, default=100)
    args = parser.parse_args()

    process_channels_file(os.path.join(args.base_folder, args.channels_file), start=args.start, num_segments=args.num_segments)

    # BASE_FILE_PATH = "/grogu/user/mhzhou/youtube-curiosity/dataset/scraping/"
    # files = [
    #     "howto100m_channels.json",
    #     "wordnet_channels.json",
    #     "tag_channels.json",
    # ]

    # process_channels_file(os.path.join(BASE_FILE_PATH, "wordnet_channels.json"), start=99, num_segments=100)
