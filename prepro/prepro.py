
import random
import numpy as np
import os
import argparse
import json
from collections import defaultdict

from libs import Tree, write_file

random.seed(1337)

def main(out_dir):

    # build tree
    data_root = args.data
    tree = Tree(data_root)
    np.save(os.path.join(out_dir, 'tree.npy'), tree)

    # find leaves and generate codeword, relation path as each node
    if args.display:
        tree.show()
    tree.gen_codeword()
    tree.gen_rel_path()

    # find nodes we want, and get codewords under these nodes
    used_nodes = {}
    for n_id, name in tree.used_nodes.items():
        used_nodes[n_id] = tree.nodes.get(name).copy()
        used_nodes[n_id].codeword = tree.get_codeword(name)
        # generate mask for internal nodes other than root node
        if n_id > 0:
            n_cw = tree.nodes.get(name).codeword
            idx = n_cw.tolist().index(1)
            used_nodes[n_id].mask = 1 - n_cw
            assert used_nodes[n_id].mask[idx] == 0
            used_nodes[n_id].mask[idx] = 1
    print('number of used nodes: {}'.format(len(used_nodes)))
    np.save(os.path.join(out_dir, 'used_nodes.npy'), used_nodes)

    # save leaf nodes
    leaf_id = {v: k for k, v in tree.leaf_nodes.items()}       # node_name: id
    print('number of classes: {}'.format(len(leaf_id)))
    np.save(os.path.join(out_dir, 'leaf_nodes.npy'), leaf_id)

    # save label at each node for each class
    node_labels = defaultdict(list)
    for k in tree.leaf_nodes.keys():
        for n_id in used_nodes.keys():
            chd_idx = np.where(used_nodes[n_id].codeword[:, k] == 1)[0]
            if len(chd_idx) > 0:
                node_labels[k].append([n_id, chd_idx[0]])

    np.save(os.path.join(out_dir, 'node_labels.npy'), node_labels)

    # load subsamples
    if args.subsample:
        fnames = json.load(open(args.subsample, 'r'))['fnames']
        split = args.subsample.split('/')[-1].replace('.json', '')
    else:
        split = 'all'

    # label data
    data = []
    for root, dirs, files in os.walk(data_root, topdown=True):
        if len(dirs) == 0:
            cls = root.split('/')[-1]
            while cls not in leaf_id:
                cls = tree.get_parent(cls)

            print('labeling {} as {} ...'.format(root, cls))
            label = leaf_id.get(cls, -1)

            if label < 0:
                raise ValueError('{} is not labeled'.format(cls))

        if args.subsample:
            for img in files:
                if img in fnames:
                    impath = os.path.join(root, img)
                    data.append((impath, label))
        else:
            for img in files:
                impath = os.path.join(root, img)
                data.append((impath, label))

    # shuffle data
    N = len(data)
    data = [(i, x) for i, x in enumerate(data)]
    random.shuffle(data)
    write_file(os.path.join(out_dir, 'gt_{}.txt'.format(split)), data)


if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        help='image data root',
    )
    parser.add_argument(
        '--out',
        type=str,
        help='output file basename',
    )
    parser.add_argument(
        '--display',
        type=bool,
        default=True,
        help='show tree hierachies',
    )
    parser.add_argument(
        '--subsample',
        type=str,
        help='subsample json file',
    )

    args = parser.parse_args()
    out_dir = os.path.join('prepro/data', '{}'.format(args.out))
    print('Data is saved to {}'.format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(out_dir)

