import argparse
import cv2
import json
import os
import torch
import torchvision
from torchvision import transforms
from diffusers import AmusedPipeline
from tqdm import tqdm, trange


def load_amused_vqgan(resolution=256, device='cuda'):
    assert resolution in (256, 512)
    pipe = AmusedPipeline.from_pretrained(
        f"amused/amused-{resolution}", variant="fp16", torch_dtype=torch.float16
    )
    pipe.vqvae.to(torch.float32)
    vqvae = pipe.vqvae
    del pipe  # free up memory
    return vqvae.to(device)


def batch_encode_imgs(vqvae, images):
    # images: list of tensors
    batch_img_tensor = images.to(vqvae.device)
    with torch.inference_mode():
        latents = vqvae.encode(batch_img_tensor).latents
        return vqvae.quantize(latents)[2][2].cpu().tolist()


def main(args):
    # load the videos we should tokenize
    files = sorted(os.listdir(args.input_dir))
    files = files[args.worker_idx::args.n_workers]
    if args.n_videos:
        files = files[:args.n_videos]
    print(f'Worker {args.worker_idx} processing {len(files)} videos')

    # read the jsonl file if it exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'worker_{args.worker_idx}.jsonl')
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        done_lines = {line['id'] for line in lines}
        files = [file for file in files if file not in done_lines]

    if len(files) == 0:
        print(f'Worker {args.worker_idx} has nothing to do')
        return

    # load vqgan
    assert args.tokenizer == 'amused' and args.resolution == 256
    vqvae = load_amused_vqgan(resolution=args.resolution)
    vqvae = torch.compile(vqvae)
    # test_recon_error(vqvae)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    with open(output_file, 'a') as f:
        for file in tqdm(files):
            video = cv2.VideoCapture(os.path.join(args.input_dir, file))
            idxs = []
            batch_frames = []
            n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv2.CAP_PROP_FPS))

            print(file)
            for i in trange(n_frames, leave=False):
                # video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()
                if not ret:
                    print('breaking')
                    break
                if i % (fps // args.fps) == 0:
                    batch_frames.append(transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    if len(batch_frames) == args.batch_size:
                        idxs.extend(batch_encode_imgs(vqvae, torch.stack(batch_frames)))
                        batch_frames = []
            video.release()
            if len(batch_frames) > 0:
                idxs.extend(batch_encode_imgs(vqvae, torch.stack(batch_frames)))
            f.write(json.dumps({'id': file.split('.mp4')[0], 'idxs': idxs}) + '\n')


def test_recon_error(vqvae):
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    img = Image.open('sharp-cat.jpeg').convert('RGB')
    img = transform(img).unsqueeze(0).to(vqvae.device)
    dtype = torch.float32
    vqvae = vqvae.to(dtype)
    with torch.inference_mode():
        latents = vqvae.encode(img.to(vqvae.device).to(dtype)).latents
        idxs = vqvae.quantize(latents)[2][2]
        recon = vqvae.decode(
            idxs,
            force_not_quantize=True,
            shape=(
                1,
                16,  # height // self.vae_scale_factor,
                16,  # width // self.vae_scale_factor,
                64,  # self.vqvae.config.latent_channels,
            ),
        ).sample.clip(0, 1)
    print(torch.mean((recon - img) ** 2).item())
    # recon_img = Image.fromarray(np.array(recon.cpu().squeeze().permute(1, 2, 0) * 255).astype(np.uint8))
    # recon_img.save('sharp-cat-recon-amuse.jpeg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='videos', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='video_tokens', help='Output directory')
    parser.add_argument('--n_videos', type=int, default=None, help='Number of videos to tokenize')
    parser.add_argument('--fps', type=int, default=3, help='Frames per second')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the work across')
    parser.add_argument('--worker_idx', type=int, default=0)

    # tokenizer kwargs
    parser.add_argument('--tokenizer', type=str, default='amused', help='Which tokenizer to use')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution of the video')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for tokenization')
    args = parser.parse_args()

    main(args)
