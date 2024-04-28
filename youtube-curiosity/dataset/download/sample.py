import csv
import cv2
import numpy as np
import random
import torch
import torchvision

from dataset.download.transcription import download_video
from dataset.download.frames import get_random_frame
from commoncrawl import YOUTUBE_VIDEOS_OUTPUT_CSV

if __name__ == "__main__":
    with open(YOUTUBE_VIDEOS_OUTPUT_CSV, "r") as f:
        reader = csv.reader(f)
        video_ids = [row[0] for row in reader]

        video_samples = random.sample(video_ids, 10)
        print("Video Samples:")
        print("\n".join(video_samples))
        print("==============")

        frame_sample_ids = random.sample(video_ids, 200)

        frames = []

        # frames = torch.load("frames.pt")
        # frame_tensor = torch.cat([torch.Tensor(frame).unsqueeze(0) for frame in frames], dim=0)

        for sample in frame_sample_ids:
            if len(frames) >= 144:
                break

            print(sample)
            download_video(sample)

            try:
                frame = get_random_frame(sample)
            except AssertionError:
                # video was not downloaded successfully
                continue

            # resize all frames to 720 x 1280
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

            frames.append(torch.Tensor(frame).unsqueeze(0))
            torch.save(frames, "frames.pt")

        frame_tensor = torch.cat(frames, dim=0)
        frame_tensor = frame_tensor.permute(0, 3, 1, 2) / 255.0

        # R and B channels are switched for some reason
        temp = frame_tensor[:, 0, :, :].clone()
        frame_tensor[:, 0, :, :] = frame_tensor[:, 2, :, :]
        frame_tensor[:, 2, :, :] = temp

        grid = torchvision.utils.make_grid(frame_tensor, nrow=12)

        torchvision.utils.save_image(grid, "../assets/frames.png")