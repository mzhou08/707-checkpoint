import subprocess
import shlex
import re

n_workers = 100
command_template = 'sbatch -J "ego4d{}" run_all.sh python scripts/tokenize_video_mp.py --input_dir /grogu/datasets/ego4d/v1/full_scale --batch_size 32 --n_workers {} --worker_idx {}'

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