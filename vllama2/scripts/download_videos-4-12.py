import argparse
import json
import os
from pytube import YouTube


def download(id, output_dir='videos'):
    url = f'https://www.youtube.com/watch?v={id}'
    yt = YouTube(url)
    try:
        yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first().download(output_path=output_dir,
                                                                                               filename=f'{id}.mp4')
        print(f'Downloaded {id}')
        return True
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print(f'Failed to download {id}')
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='videos', help='Output directory')
    parser.add_argument('--n_videos', type=int, default=1, help='Number of videos to download')
    parser.add_argument('--source_json', type=str, default='data/videos.json', help='Path to JSON file with video IDs')

    args = parser.parse_args()

    # load json
    with open(args.source_json) as f:
        ids = json.load(f)

    downloaded = 0
    i = 0
    while downloaded < args.n_videos and i < len(ids):
        id = ids[i]
        if os.path.exists(os.path.join(args.output_dir, f'{id}.mp4')):
            print(f'Skipping {id}')
            i += 1
            continue
        i += 1
        if download(id, args.output_dir):
            downloaded += 1
