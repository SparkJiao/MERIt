import collections
import glob
import pickle
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

    all_data = collections.defaultdict(list)
    for file in tqdm(input_files):
        # data = json.load(open(file))
        data = pickle.load(open(file, "rb"))
        for key in data:
            all_data[key].extend(data[key])

    dev_num = {key: int(len(all_data[key]) * args.dev_ratio) for key in all_data}
    print(f"Dev num: {dev_num}")
    dev_ids = {key: set(random.sample(range(len(all_data[key])), dev_num[key])) for key in all_data}

    if args.dev_ratio > 0:
        train_data = collections.defaultdict(list)
        dev_data = collections.defaultdict(list)
        for key in all_data:
            for t_id, t in enumerate(tqdm(all_data[key], desc=f'splitting data in {key}', total=len(all_data))):
                if t_id in dev_ids[key]:
                    dev_data[key].append(t)
                else:
                    train_data[key].append(t)
    else:
        train_data = all_data
        dev_data = None

    split_num = {key: len(train_data[key]) // args.group for key in all_data}
    print(f"Split num: {split_num}")
    for idx in range(args.group):
        if idx < args.group - 1:
            sub_data = {key: train_data[key][(idx * split_num[key]): ((idx + 1) * split_num[key])] for key in all_data}
        else:
            sub_data = {key: train_data[key][(idx * split_num[key]):] for key in all_data}
        # json.dump(sub_data, open(args.output_file.replace('.json', f'.train.{idx}.json'), 'w'))
        pickle.dump(sub_data, open(args.output_file.replace('.pkl', f'.train.{idx}.pkl'), 'wb'))

    if dev_data is not None:
        # json.dump(dev_data, open(args.output_file.replace('.json', '.dev.json'), 'w'))
        pickle.dump(dev_data, open(args.output_file.replace('.pkl', '.dev.pkl'), 'wb'))
