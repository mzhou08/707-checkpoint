# Loads the tokenized ego4d videos and combines them into a single file.
# Splits them into lines of 2048 tokens each.

import json
import os
from tqdm import tqdm


def main():
    seq_len = 2048
    output_dir = 'video_tokens'
    mappings = {
        'video_tokens/ego4d_combined.jsonl': range(99),
        'video_tokens/ego4d_val.jsonl': [99]
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
    main()
