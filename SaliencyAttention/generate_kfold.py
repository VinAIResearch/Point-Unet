import pickle
import numpy as np
import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='training data path', default="./BraTS2020/training/")
    parser.add_argument('--out', help="output path", default="./3folds")
    parser.add_argument('--nfolds', type=int, help="number of folds", default=3)
    args = parser.parse_args()
    data = {}

    n_folds = args.nfolds

    HGG_filenames = glob.glob(args.data+"HGG/*")
    print(len(HGG_filenames))
    val_length_HGG = len(HGG_filenames) // n_folds
    np.random.shuffle(HGG_filenames)

    folds = []
    for i in range(n_folds):
        if i < n_folds - 1:
            folds.append(HGG_filenames[(val_length_HGG * i):(val_length_HGG * (i + 1))])
        else:
            folds.append(HGG_filenames[(val_length_HGG * i):])

    for i in range(n_folds):
        data['fold{}'.format(i)] = {}
        data['fold{}'.format(i)]['val'] = folds[i]

        data['fold{}'.format(i)]['training'] = []
        for j in range(n_folds):
            if j == i:
                continue
            else:
                data['fold{}'.format(i)]['training'] +=  folds[j]
        print(len(data['fold{}'.format(i)]['val']), len(data['fold{}'.format(i)]['training']))

    with open(args.out+".pkl", 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)




