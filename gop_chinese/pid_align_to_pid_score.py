from os.path import join, abspath, dirname
import json

file_dir = dirname(abspath(__file__))
project_root_dir = dirname(file_dir)

pid_align = join(project_root_dir, 'tdnn_align', 'lang', 'phones.txt')
pid_score = join(project_root_dir, 'tdnn_score', 'lang', 'phones.txt')
out_file = join(file_dir, 'pid_align_to_pid_score.json')

p_align = {}
with open(pid_align) as f:
    for line in f:
        name, i = line.replace('\n', '').split()
        p_align[name] = int(i)

p_score = {}
with open(pid_score) as f:
    for line in f:
        name, i = line.replace('\n', '').split()
        p_score[name] = int(i)

pid_align_to_pid_score = {}
for name, i in p_align.items():
    if '_' not in name:
        pid_align_to_pid_score[i] = p_score[name]
        continue
    phone = '_'.join(name.split('_')[:-1])
    pid_align_to_pid_score[i] = p_score[phone]

json.dump(pid_align_to_pid_score, open(out_file, 'w'))
