import torch
import json

def iter_batches(split='train', batch_size=256, device='cuda'):
    if split == 'train':
        file = '/home/mhzhou/vllama2/video_tokens/videos_train.jsonl'
    elif split == 'val':
        file = '/home/mhzhou/vllama2/video_tokens/videos_val.jsonl'
    else:
        raise ValueError(f"Split {split} not supported!")

    batch = []
    while True:
        with open(file, 'r') as f:
            for line in f:
                line_dict = json.loads(line)
                batch.append(line_dict['tokens'])
                if len(batch) == batch_size:
                    y = torch.tensor(batch, dtype=torch.long).to(device)
                    x = torch.cat([torch.full((batch_size, 1), 8192, dtype=torch.long, device=device),
                                   y[:, :-1]], dim=1)
                    yield x, y
                    batch = []