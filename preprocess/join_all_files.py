import glob
import json
import random
from argparse import ArgumentParser

from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--dev_ratio', type=float, default=0.05)
    parser.add_argument('--group', type=int, default=1)

    args = parser.parse_args()

    input_files = list(glob.glob(args.input_file))
    print(input_files)

    all_data = []
    for file in tqdm(input_files):
        all_data.extend(json.load(open(file)))

    dev_num = int(len(all_data) * args.dev_ratio)
    dev_ids = set(random.sample(range(len(all_data)), dev_num))

    if args.dev_ratio > 0:
        train_data = []
        dev_data = []
        for t_id, t in enumerate(tqdm(all_data, desc='splitting data', total=len(all_data))):
            if t_id in dev_ids:
                dev_data.append(t)
            else:
                train_data.append(t)
    else:
        train_data = all_data
        dev_data = None

    split_num = len(train_data) // args.group
    for idx in range(args.group):
        if idx < args.group - 1:
            sub_data = train_data[(idx * split_num): ((idx + 1) * split_num)]
        else:
            sub_data = train_data[(idx * split_num):]
        json.dump(sub_data, open(args.output_file.replace('.json', f'.train.{idx}.json'), 'w'))

    if dev_data is not None:
        json.dump(dev_data, open(args.output_file.replace('.json', '.dev.json'), 'w'))
