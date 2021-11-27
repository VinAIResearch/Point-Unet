from os.path import join
from RandLANet import Network
from testBraTS import ModelTester
from helper_ply import read_ply
from helper_tool import ConfigBraTS as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os
import random

path_train_ID = "../dataset/BraTS2020/train_BraTS20.txt"
with open(path_train_ID) as f:
    content = f.readlines()
train_IDs = [x.strip() for x in content] 

path_val_ID = "../dataset/BraTS2020/valOffline_BraTS20.txt"
with open(path_val_ID) as f:
    content = f.readlines()
val_IDs = [x.strip() for x in content] 



class BraTS:
    def __init__(self, Mode):
        self.name = 'BraTS20'
        self.path = cfg.data_PC_path
        self.label_to_names = {0: '0',
                               1: '1',
                               2: '2',
                               3: '3' }
        
        self.num_point = cfg.num_points
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])
        self.Mode = Mode
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.train_IDs = train_IDs
        self.val_IDs = val_IDs
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.input_xyz_origin = {'training': [], 'validation': []}
        self.weight = {'training': [], 'validation': []}
        self.distribute = np.array([0.0,0.0,0.0,0.0])
        self.max_tumor = 0
        self.load_sub_sampled_clouds(cfg.sub_grid_size)




    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path,'input0.01')
        for i, IDs in enumerate(self.all_files):
            file_path = IDs
            cloud_name = file_path.split('/')[-1][:-4]


            if self.Mode == "train":
                if cloud_name in self.train_IDs:
                    cloud_split = 'training'
                else:
                    cloud_split = 'validation'
                self.input_names[cloud_split] += [file_path]
            else:
                if cloud_name  in self.val_IDs:
                    cloud_split = 'validation'
                    xyz_file = os.path.join(tree_path,cloud_name+"_xyz_origin.npy")
                    xyz_origin = np.load(xyz_file)
                    self.input_xyz_origin[cloud_split] += [xyz_origin]
                    self.input_names[cloud_split] += [file_path]




    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = int(len(self.input_names["training"])/cfg.batch_size)*cfg.batch_size

        elif split == 'validation':
            num_per_epoch = int(len(self.input_names["validation"])/cfg.val_batch_size)*cfg.val_batch_size

        def spatially_regular_gen():
            # Generator loop
            print("Status: ",split, "  .Num per epoch: ", num_per_epoch)
            for i in range(num_per_epoch):
                cloud_idx = i
                file_path = self.input_names[split][i]
                full_ply_file = join(file_path)

                data = read_ply(full_ply_file)
                queried_pc_xyz =  np.vstack((data['x'], data['y'], data['z'])).T
                sub_colors = np.vstack((data['t1ce'], data['t1'], data['flair'], data['t2'])).T
                sub_labels = data['class']


                masks = sub_labels
                all_label = masks
                none_tumor = list(np.where(all_label == 0)[0])
                tumor = list(np.where(all_label > 0)[0])
                queried_idx = tumor + random.sample(none_tumor, k=cfg.num_points - len(tumor))
                queried_idx = np.array(queried_idx)


                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = queried_pc_xyz[queried_idx]
                # queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = sub_colors[queried_idx]
                queried_pc_labels = sub_labels[queried_idx]

                (unique, counts) = np.unique(queried_pc_labels, return_counts=True)
                frequencies = np.asarray((unique, counts)).T
                # print("queried_pc_labels ",len(queried_pc_labels), frequencies)
                
                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 4], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test')
    parser.add_argument('--n_epoch', type=int, default=100, help='number of epoch')
    parser.add_argument('--logdir', type=str, default='./PointSegment/model_logs/BraTS20', help='path to the log directory')
    parser.add_argument('--data_PC_path', type=str, default='./dataset/BraTS2020/PC_data', help='path to the point cloud data')
    parser.add_argument('--checkpoint_path', type=str, default='./PointSegment/model_logs/BraTS20/snapshots/snap-8261', help='path to the checkpoint')
    parser.add_argument('--results_path', type=str, default='../dataset/BraTS2020/predict_npy', help='path to save segmentation results')
    FLAGS = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    cfg.saving_path = FLAGS.logdir
    if not os.path.exists(cfg.saving_path):
        os.mkdir(cfg.saving_path)
    cfg.train_sum_dir = cfg.saving_path+'/train_log/'
    cfg.log_file = cfg.saving_path + "/train_summary.txt" 
    cfg.data_PC_path = FLAGS.data_PC_path
    cfg.max_epoch = FLAGS.n_epoch
    chosen_snap = FLAGS.checkpoint_path
    output_save = FLAGS.results_path
    if not os.path.exists(output_save):
        os.makedirs(output_save)



    Mode = FLAGS.mode

    dataset = BraTS(Mode)
    dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        tester = ModelTester(model, dataset, output_save, restore_snap=chosen_snap)
        tester.test(model, dataset)
    else:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels = flat_inputs[21]
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])


# Traning
# python3 PointSegment/runBraTS.py --gpu 0 --mode train --n_epoch 100 --logdir ./PointSegment/model_logs/BraTS20 --data_PC_path ../dataset/BraTS2020
# Predict
# python3 PointSegment/runBraTS.py --gpu 0 --mode test --data_PC_path ../dataset/BraTS2020 --checkpoint_path /vinai/vuonghn/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/0.01_float/BraTS20_0.01_float/Point-Unet/output/BraTS20_CE/snapshots/snap-8261 --results_path ../dataset/BraTS2020/predict_npy