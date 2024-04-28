"""
Takes in video IDs of scraped videos and writes tokenized videos to a JSONL file.
"""

import subprocess
import shlex
import re
import json
import os
from tqdm import tqdm


def encode_custom_videos():
    n_workers = 100
    command_template = 'sbatch -J "videos{}" run_all.sh python scripts/tokenize_video_mp.py --input_dir /grogu/user/mhzhou/youtube-curiosity/videos/ --batch_size 32 --n_workers {} --worker_idx {}'

    commands = []
    for i in range(n_workers):
        command = command_template.format(i, n_workers, i)
        commands.append(command)
        print(command)

    print_command = False
    if not print_command:
        for command in commands:
            output = subprocess.check_output(shlex.split(command)).decode('utf8')
            job_names = list(re.findall(r'\d+', output))
            assert (len(job_names) == 1)
            print(job_names)

def main():
    seq_len = 2048
    output_dir = 'video_tokens'
    mappings = {
        'video_tokens/videos_train.jsonl': range(99),
        'video_tokens/videos_val.jsonl': [99]
    }
    for combined_file, worker_idxs in mappings.items():
        for worker_idx in tqdm(worker_idxs):
            output_file = os.path.join(output_dir, f'worker_{worker_idx}.jsonl')
            count = 0
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                lines = [json.loads(line) for line in lines]
                # write the filtered lines back to the file
                with open(combined_file, 'a') as f:
                    for line in lines:
                        for i in range(len(line['tokens']) // seq_len):
                            to_write = json.dumps(
                                dict(id=line['id'], tokens=line['tokens'][i * seq_len: (i + 1) * seq_len]))
                            f.write(to_write + '\n')
                            count += 1
                print(f'Worker {worker_idx}: {count} sequences written')


if __name__ == '__main__':
    encode_custom_videos()
    main()