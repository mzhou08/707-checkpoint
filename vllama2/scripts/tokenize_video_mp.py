import argparse
import cv2
import json
import multiprocessing as mp
import os
import os.path as osp
import torch
import torchvision
from torchvision import transforms
from tokenize_video import load_amused_vqgan, batch_encode_imgs
from collections import defaultdict


# Function to process video frames
def process_video(name, video_path, frame_queue, batch_size=32, target_fps=3, res=256):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
    ])
    cap = cv2.VideoCapture(video_path)
    batch_frames = []
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // target_fps

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            frame_queue.put((name, torch.stack(batch_frames)))
            frame_queue.put((name, None))  # Termination signal
            cap.release()
            break
        if i % frame_interval == 0:
            batch_frames.append(transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(batch_frames) == batch_size:
                # print('Sending from', name, f'{i}/{n_frames}')
                frame_queue.put((name, torch.stack(batch_frames)))
                batch_frames = []
    if len(batch_frames) > 0:
        frame_queue.put((name, torch.stack(batch_frames)))
    frame_queue.put((name, None))  # Termination signal
    cap.release()


# Function for central process to dequeue frames and encode them
def encoder_process(frame_queue, output_file, tokenizer, resolution):
    # load tokenizer model
    assert tokenizer == 'amused' and resolution == 256
    vqvae = load_amused_vqgan(resolution=resolution)
    vqvae = torch.compile(vqvae)
    print('Successfully loaded tokenizer model')

    name_to_idxs = defaultdict(list)
    with open(output_file, 'a') as f:
        while True:
            next_item = frame_queue.get()  # Get frame from the queue
            if next_item is None:  # Exit condition
                del vqvae
                return
            else:
                name, frames = next_item
                # print(name, (time.time() - start) / count)
            if frames is None:
                # done with this video, write tokens to file
                f.write(json.dumps({'id': name, 'tokens': name_to_idxs[name]}) + '\n')
                name_to_idxs.pop(name)
                print(f'Done with {name}')
                continue
            encoded_frame = batch_encode_imgs(vqvae, frames)
            name_to_idxs[name].extend(encoded_frame)



def main(args):
    # load the videos we should tokenize
    files = sorted([f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))])
    print(f'Found {len(files)} videos')
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
        files = [file for file in files if file.split('.mp4')[0] not in done_lines]

    if len(files) == 0:
        print(f'Worker {args.worker_idx} has nothing to do')
        return

    m = mp.Manager()
    frame_queue = m.Queue()

    # Start the central process
    central_p = mp.Process(target=encoder_process, args=(frame_queue, output_file, args.tokenizer, args.resolution))
    central_p.start()

    video_args = [(f.split('.mp4')[0],
                   osp.join(args.input_dir, f),
                   frame_queue,
                   args.batch_size,
                   args.fps,
                   args.resolution) for f in files]
    with mp.Pool(args.n_mp_workers) as pool:
        pool.starmap(process_video, video_args)

    # Send termination signal to the central process
    frame_queue.put(None)
    central_p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='videos', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='video_tokens', help='Output directory')
    parser.add_argument('--n_videos', type=int, default=None, help='Number of videos to tokenize')
    parser.add_argument('--fps', type=int, default=3, help='Frames per second')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the work across')
    parser.add_argument('--n_mp_workers', type=int, default=8, help='Number of workers to split the work across')
    parser.add_argument('--worker_idx', type=int, default=0)

    # tokenizer kwargs
    parser.add_argument('--tokenizer', type=str, default='amused', help='Which tokenizer to use')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution of the video')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for tokenization')
    args = parser.parse_args()

    main(args)
