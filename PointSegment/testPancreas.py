from os import makedirs
from os.path import exists, join
from helper_ply import write_ply
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time
import nibabel
import numpy as np
import random
import os
import pickle
from scipy import ndimage
import copy



def preprocess_label(label):
    bachground = label == 0
    pancreas = label == 1  
    return np.array([bachground, pancreas], dtype=np.uint8)
def dice_coefficient(truth, prediction):
    if (np.sum(truth) + np.sum(prediction)) == 0:
        return 1
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def save_to_nii(im, filename, outdir="", mode="image", system="nibabel"):
    """
    Save numpy array to nii.gz format to submit
    im: 3d numpy array ex: [155, 240, 240]
    """
    im = im.astype(np.uint8)
    if system == "sitk":
        if mode == 'label':
            img = sitk.GetImageFromArray(im.astype(np.uint8))
        else:
            img = sitk.GetImageFromArray(im.astype(np.float32))
        if not os.path.exists("./{}".format(outdir)):
            os.mkdir("./{}".format(outdir))
        sitk.WriteImage(img, "./{}/{}.nii.gz".format(outdir, filename))
    elif system == "nibabel":
        img = np.moveaxis(im, 0, -1)
        OUTPUT_AFFINE = np.array(
                [[ -1,-0,-0,-0],
                 [ -0,-1,-0,239],
                 [  0, 0, 1,0],
                 [  0, 0, 0,1]])
        if mode == 'label':
            img = nibabel.Nifti1Image(img.astype(np.uint8), OUTPUT_AFFINE)
        else:
            img = nibabel.Nifti1Image(img.astype(np.float32), OUTPUT_AFFINE)
        if not os.path.exists("./{}".format(outdir)):
            os.mkdir(outdir)
        nibabel.save(img, "./{}/{}.nii.gz".format(outdir, filename))
    else:
        img = np.rot90(im, k=2, axes= (1,2))
        OUTPUT_AFFINE = np.array(
                [[0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]])
        if mode == 'label':
            img = nibabel.Nifti1Image(img.astype(np.uint8), OUTPUT_AFFINE)
        else:
            img = nibabel.Nifti1Image(img.astype(np.float32), OUTPUT_AFFINE)
        if not os.path.exists("./{}".format(outdir)):
            os.mkdir("./{}".format(outdir))
        nibabel.save(img, "./{}/{}.nii.gz".format(outdir, filename))

def point2prod(list_point_prod,list_xyz,ID,volume_shape,root_save):
    
    list_xyz = np.load(list_xyz)

    
    volume = np.zeros(volume_shape)
    

    for i in range(len(list_point_prod)):
        volume[list_xyz[i][2]][list_xyz[i][0]][list_xyz[i][1]] = list_point_prod[i]

    volume = np.moveaxis(volume, 1, 2)
    path_save  = os.path.join(root_save,ID+".npy")
    print("path_save ", path_save)
    np.save(path_save, volume)



def point2volume(list_point,list_xyz):
    volume = np.zeros((155,240,240))
    for i in range(len(list_point)):
        volume[list_xyz[i][2]][list_xyz[i][0]][list_xyz[i][1]] = list_point[i]
    volume[volume==3]=4
    print("np.unique(volume) ",np.unique(volume) )
    return volume

def point2volume_and_save(list_point,list_xyz,ID):
    volume3D = point2volume(list_point,list_xyz)
    save_to_nii(volume3D,ID,outdir) 

def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)

class ModelTester:
    def __init__(self, model, dataset,root_save , restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open('log_test_BraTS20_dense.txt', 'a')
        self.root_save = root_save
        self.Dice_test = []

        if not os.path.exists(self.root_save):
            os.makedirs(self.root_save)



        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)

        print("dataset.input_names['validation'] ", len(dataset.input_names['validation']))

        # Initiate global prediction over all test clouds
        # self.test_probs = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
        #                    for l in dataset.input_names['validation']]
    def test(self, model, dataset, num_votes=100):
        num_votes =1
        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1
        

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test/testPancreas', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        count_number = 0
        while count_number < len(dataset.all_files):
            

            ops = (self.prob_logits,
                    model.labels,
                    model.inputs['input_inds'],
                    model.inputs['cloud_inds'],
                    )
            
            stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})

            
            correct = np.sum(np.argmax(stacked_probs, axis=1) == stacked_labels)

            
            truth_seg = preprocess_label(stacked_labels)
            pred_seg = preprocess_label(np.argmax(stacked_probs, axis=1))
           

            dice = dice_coefficient(truth_seg[1],pred_seg[1])

            print(dataset.input_names['validation'][cloud_idx[0][0]], dice)
            
            self.Dice_test.append(dice)
           

            acc = correct / float(np.prod(np.shape(stacked_labels)))
            
            stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, len(stacked_labels),
                                                        model.config.num_classes])

            probs = stacked_probs[0, :, :]

            
            point2prod(probs,dataset.input_xyz_origin['validation'][cloud_idx[0][0]],dataset.input_names['validation'][cloud_idx[0][0]],dataset.xyz_shape['validation'][cloud_idx[0][0]],self.root_save)

            count_number+=1
        print('dice mean', np.mean(self.Dice_test))


            