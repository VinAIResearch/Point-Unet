import os
import shutil

if __name__ == '__main__':
    ROOT = '/home/ubuntu/datasets/BraTS2020/training/HGG'
    with open('val_IDs_fold1_82_182.txt') as f:
        for l in f:
            folder_name = l.rstrip()
            folder_path = os.path.join(ROOT, folder_name)
            shutil.rmtree(folder_path)
    print(len(os.listdir(ROOT)))
