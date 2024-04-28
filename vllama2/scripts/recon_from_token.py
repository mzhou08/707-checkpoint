import argparse
import cv2
import json
import os
import torch
import numpy as np
from tokenize_video import load_amused_vqgan


def images_to_video(images, video_name, fps):
    # images: np array of shape (n_frames, height, width, 3)
    _, height, width, _ = images.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        # convert image to bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def main(args):
    # load tokenizer model
    assert args.tokenizer == 'amused' and args.resolution == 256
    vqvae = load_amused_vqgan(resolution=args.resolution)
    vqvae = torch.compile(vqvae)
    print('Successfully loaded tokenizer model')

    # load lines
    lines = dict()
    with open(args.token_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            lines[line['id']] = line['tokens']
            if len(lines) == args.n_videos:
                break

    # recon
    os.makedirs(args.output_dir, exist_ok=True)
    for name, tokens in lines.items():
        recons = []
        with torch.inference_mode():
            for i in range(0, len(tokens), 256 * args.batch_size):
                tokens_on_dev = torch.tensor(tokens[i: i + 256 * args.batch_size]).to(vqvae.device).reshape(-1, 256)
                recon = vqvae.decode(
                    tokens_on_dev,
                    force_not_quantize=True,
                    shape=(
                        len(tokens_on_dev),
                        16,  # height // self.vae_scale_factor,
                        16,  # width // self.vae_scale_factor,
                        64,  # self.vqvae.config.latent_channels,
                    ),
                ).sample.clip(0, 1)
                recons.append(recon.cpu().numpy())

        recons = (np.concatenate(recons, axis=0).transpose(0, 2, 3, 1) * 256).astype(np.uint8)
        images_to_video(recons, f'{args.output_dir}/{name}.mp4', args.fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_file', type=str, required=True, help='Where to load saved tokens from')
    parser.add_argument('--output_dir', type=str, default='video_recons', help='Output directory')
    parser.add_argument('--n_videos', type=int, default=1, help='Number of videos to reconstruct')
    parser.add_argument('--fps', type=int, default=3, help='Frames per second')

    # tokenizer kwargs
    parser.add_argument('--tokenizer', type=str, default='amused', help='Which tokenizer to use')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution of the video')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for tokenization')
    args = parser.parse_args()

    main(args)
