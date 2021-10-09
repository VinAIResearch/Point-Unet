from os.path import join
from RandLANet import Network
from test_Pancreas import ModelTester
from helper_ply import read_ply
from helper_tool import ConfigPancreas as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os
import random
import nibabel as nib



path_pancreas = "/vinai/vuonghn/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas_v1/full_size_dilation_attention/"
original_volume = "/vinai/vuonghn/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas-CT_processed/Pancreas-CT_processed_v1/ct"

class Pancreas:
    def __init__(self, Mode):
        self.name = 'Pancreas'
        self.path = path_pancreas
        self.label_to_names = {0: '0',
                               1: '1'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])
        self.Mode = Mode
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}

        self.ply_data = {'training': [], 'validation': []}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.path_ply = {'training': [], 'validation': []}
        self.path_xyz = {'training': [], 'validation': []}
        self.xyz_shape = {'training': [], 'validation': []}
        self.input_xyz_origin = {'training': [], 'validation': []}
        self.weight = {'training': [], 'validation': []}
        self.distribute = np.array([0.0,0.0,0.0,0.0])
        self.max_tumor = 0
        self.fold = 3
        self.all_files = self.get_all_file()
 
        self.load_sub_sampled_clouds(cfg.sub_grid_size)



    def get_all_file(self):

        if self.Mode == "train":
            all_files = glob.glob(join(path_pancreas, 'original_ply', '*.ply'))
        else:
            all_files = []
            for i in range(self.fold, 83,4):
                if i ==0 or i > 82:
                    continue
                for j in range(8):
                    ID = str(i).zfill(4)+"_loop_"+str(j)+".ply"
                    file_ID = os.path.join(path_pancreas,"original_ply",ID)
                    all_files.append(file_ID)
        return all_files



    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path,'input0.01')

        max_mask = 0
        
        for i, file_path in enumerate(self.all_files):
            
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:4]
            name_ID = file_path.split('/')[-1][:-4]
            xyz_file = os.path.join(tree_path,name_ID.split('_loop')[0]+"_xyz_origin_loop_"+name_ID.split('_loop_')[1]+".npy")
            print("cloud_name ", cloud_name)
            if self.Mode == "train":
                if int(cloud_name)%4==self.fold:
                    self.path_ply["validation"] += [file_path]
                    self.input_names["validation"] += [name_ID]
                else:
                    self.path_ply["training"] += [file_path]
                    self.input_names["training"] += [name_ID]
                print("loading.... ", name_ID)
                

            else:
                path_volume3D = os.path.join(original_volume,"PANCREAS_"+cloud_name+".nii.gz")
                volume_shape = np.asanyarray(nib.load(path_volume3D).dataobj).shape
                volume_shape = (volume_shape[2],volume_shape[0],volume_shape[1],2)  
                self.path_ply["validation"] += [file_path]
                self.input_names["validation"] += [name_ID]
                self.input_xyz_origin["validation"] += [xyz_file]
                self.xyz_shape["validation"] += [volume_shape]
                print("loading.... ", name_ID)
        print("train ", len(self.path_ply["training"]))
        print("val ", len(self.path_ply["validation"]))

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = int(len(self.path_ply["training"])/cfg.batch_size)*cfg.batch_size

        elif split == 'validation':
            num_per_epoch = int(len(self.path_ply["validation"])/cfg.val_batch_size)*cfg.val_batch_size

        def spatially_regular_gen():
            # Generator loop
            print("Fold : ", self.fold, "Number input point : ",read_ply(join(self.path_ply[split][0]))['class'].shape )
            for i in range(num_per_epoch):
                cloud_idx = i
                file_path = self.path_ply[split][i]
                full_ply_file = join(file_path)

                # start_time = time.time()

                # data = self.ply_data[split][i]
                data = read_ply(full_ply_file)

                queried_pc_xyz =  np.vstack((data['x'], data['y'], data['z'])).T
                sub_colors = np.vstack((data['value']))
                sub_labels = data['class']
                queried_idx = np.arange(len(sub_labels))
                queried_pc_colors = queried_pc_xyz
                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           sub_colors.astype(np.float32),
                           sub_labels,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    Mode = FLAGS.mode

    test_area = FLAGS.test_area
    dataset = Pancreas(Mode)
    dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)

    
        chosen_snap = "/vinai/vuonghn/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas_v1/full_size_dilation_attention/log_dice_loss/fold3/snapshots/snap-497"
        root_save = "/vinai/vuonghn/Research/3D_Med_Seg/Point-Unet/dataset/Pancreas/"
        if not os.path.exists(root_save):
            os.mkdir(root_save)
        print("root_save 0",root_save)
        tester = ModelTester(model, dataset,root_save, restore_snap=chosen_snap)
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
    