from os.path import join
from RandLANet import Network
from test_pancreas import ModelTester
from helper_ply import read_ply
from helper_tool import ConfigPancreas as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os
import random


path_pancreas = "./Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas/mean_std"
all_files = glob.glob(join(path_pancreas, 'original_ply', '*.ply'))






class Pancreas:
    def __init__(self, test_area_idx):
        self.name = 'Pancreas'
        self.path = path_pancreas
        self.label_to_names = {0: '0',
                               1: '1'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = all_files
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        # self.train_IDs = train_IDs
        # self.val_IDs = val_IDs
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.input_xyz_origin = {'training': [], 'validation': []}
        self.weight = {'training': [], 'validation': []}
        self.distribute = np.array([0.0,0.0,0.0,0.0])
        self.max_tumor = 0
        self.fold = 0

        self.load_sub_sampled_clouds(cfg.sub_grid_size)




    def load_sub_sampled_clouds(self, sub_grid_size):
        # tree_path = join(self.path,'input')
        tree_path = join(self.path,'input_0.01')

        max_mask = 0
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            if int(cloud_name)%4==self.fold:
                self.input_names["validation"] += [file_path]
            else:
                self.input_names["training"] += [file_path]






    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = int(len(self.input_names["training"])/cfg.batch_size)*cfg.batch_size
            # num_per_epoch = 1
        elif split == 'validation':
            num_per_epoch = int(len(self.input_names["validation"])/cfg.val_batch_size)*cfg.val_batch_size
            # num_per_epoch = 1
        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):
                cloud_idx = i
                file_path = self.input_names[split][i]
                full_ply_file = join(file_path)
                start_time = time.time()
                # print("read & process data")
                data = read_ply(full_ply_file)
                queried_pc_xyz =  np.vstack((data['x'], data['y'], data['z'])).T

                sub_colors = data['value'] 
                sub_colors = np.expand_dims(sub_colors, axis=1)
                sub_labels = data['class']

                
                all_label = sub_labels
                none_tumor = list(np.where(all_label == 0)[0])
                tumor = list(np.where(all_label > 0)[0])
                queried_idx = tumor + random.sample(none_tumor, k=cfg.num_points - len(tumor))
                queried_idx = np.array(queried_idx)


 
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = queried_pc_xyz[queried_idx]
                # queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = sub_colors[queried_idx]
                queried_pc_labels = sub_labels[queried_idx]
                # print("time process ", time.time()- start_time)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 1], [None], [None], [None])
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
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    Mode = FLAGS.mode

    test_area = FLAGS.test_area
    dataset = Pancreas(test_area)
    dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)

        if FLAGS.model_path is not 'None':
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            # logs = np.sort([os.path.join('./Model_log/BraTS20', f) for f in os.listdir('results') if f.startswith('Log')])
            # chosen_folder = logs[-1]
            # snap_path = join(chosen_folder, 'snapshots')
            # snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            # chosen_step = np.sort(snap_steps)[-1]
            # chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        chosen_snap = "./Research/Brain_Point/RandLA-Net/Model_log/normalize_xyz/0.01_float/BraTS18_0.01_float/Point-Unet/output/BraTS18_dense_log/snapshots/snap-11001"

        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
    else:
        ##################
        # Visualize data #
        ##################

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
    