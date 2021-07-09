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

    args = parser.parse_args()

    input_files = list(glob.glob(args.input_file))
    print(input_files)

    all_data = []
    for file in tqdm(input_files):
        all_data.extend(json.load(open(file)))

    dev_num = int(len(all_data) * args.dev_ratio)
    dev_ids = set(random.sample(range(len(all_data)), dev_num))

    train_data = []
    dev_data = []
    for t_id, t in enumerate(tqdm(all_data, desc='splitting data', total=len(all_data))):
        if t_id in dev_ids:
            dev_data.append(t)
        else:
            train_data.append(t)

    json.dump(train_data, open(args.output_file.replace('.json', '.train.json'), 'w'))
    json.dump(dev_data, open(args.output_file.replace('.json', '.dev.json'), 'w'))
