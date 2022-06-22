from os.path import join, abspath, dirname
from typing import Tuple
import json

file_dir = dirname(abspath(__file__))

phone2same_tone_path = join(file_dir, 'phone2same_tone.json')
phone2same_base_path = join(file_dir, 'phone2same_base.json')
phones_path = join(file_dir, 'phones.txt')


def get_base_tone_from_name(name: str) -> Tuple[str, int]:
    """:return 0 -> 4 tones, -1 if phone is not toned"""
    # <base>_<tone>
    if len(name) > 2 and name[-1].isnumeric() and name[-2] == '_':
        return name[:-2], int(name[-1])
    else:
        return '', -1


if __name__ == '__main__':
    with open(phones_path, 'r') as f:
        phone_names2ids = {line.split()[0]: int(line.split()[1]) for line in f}
    n_phones = len(phone_names2ids)

    phone2same_tone = {i: [] for i in range(n_phones + 1)}
    phone2same_base = {i: [] for i in range(n_phones + 1)}

    # toned phones -> phones with the same base
    bases = []
    for pname, pid in phone_names2ids.items():
        base, tone = get_base_tone_from_name(pname)
        if tone != -1:
            bases.append(base)
            phone2same_base[pid] += [phone_names2ids[f'{base}_{t}'] for t in range(5)]
    json.dump(phone2same_base, open(phone2same_base_path, 'w'), indent=2)

    # toned phones -> phones with the same tone
    bases = list(set(bases))
    for pname, pid in phone_names2ids.items():
        _, tone = get_base_tone_from_name(pname)
        if tone != -1:
            phone2same_tone[pid] += [phone_names2ids[f'{b}_{tone}'] for b in bases]
    json.dump(phone2same_tone, open(phone2same_tone_path, 'w'), indent=2)

    # get (phones -> base and tone) and (base and tone -> phones)
    phone2base_tone = {}
    base_tone2phone = {}
    for pname, pid in phone_names2ids.items():
        base, tone = get_base_tone_from_name(pname)
        if tone != -1:
            base_i = phone_names2ids[f'{base}_0']
            phone2base_tone[pid] = [base_i, tone]
            base_tone2phone[f'{base_i}_{tone}'] = pid
    json.dump(phone2base_tone, open('phone2base_tone.json', 'w'), indent=2)
    json.dump(base_tone2phone, open('base_tone2phone.json', 'w'), indent=2)
