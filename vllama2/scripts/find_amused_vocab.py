# Finds the total VQGAN number of tokens
# Can be stopped early -- quickly becomes clear what the right number is.
import json
from tqdm import tqdm


def main():
    f = 'video_tokens/ego4d_combined.jsonl'
    max_i = 0
    min_i = 1000
    i = 0
    with open(f, 'r') as f:
        for line in tqdm(f):
            i += 1
            line_dict = json.loads(line)
            tokens = line_dict['tokens']
            max_i = max(max_i, max(tokens))
            min_i = min(min_i, min(tokens))
            if i % 10000 == 0:
                print(min_i, max_i)

    print(max_i, min_i)
if __name__ == '__main__':
    main()
