import json
import os
from tqdm import trange


def main():
    # read the jsonl file if it exists
    output_dir = 'video_tokens'
    for worker_idx in trange(100):
        output_file = os.path.join(output_dir, f'worker_{worker_idx}.jsonl')
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                lines = f.readlines()
            filtered_lines = []
            existing_ids = set()
            for line in lines:
                line = json.loads(line)
                if line['id'] not in existing_ids:
                    filtered_lines.append(line)
                    existing_ids.add(line['id'])
            # write the filtered lines back to the file
            with open(output_file, 'w') as f:
                for line in filtered_lines:
                    f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
