# /opt/homebrew/Caskroom/miniforge/base/envs/tensorflow/bin/python /Users/ducanhnguyen/Documents/mydeepconcolic/src/c_code/model_prediction_for_external.py '/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/datatest/proposal/mnist_simple/out(K=0.0,target-1)'

import pandas as pd
import os
import glob
import sys
import csv
import numpy as np
from pathlib import Path

sys.path.append('/Users/ducanhnguyen/Documents/mydeepconcolic/')
sys.path.append('/Users/ducanhnguyen/Documents/mydeepconcolic/src')
sys.path.append('/Users/ducanhnguyen/Documents/mydeepconcolic/src/utils')
sys.path.append('/Users/ducanhnguyen/Documents/mydeepconcolic/src/ae_attack_border')
sys.path.append('/Users/ducanhnguyen/Documents/mydeepconcolic/src/c_code')
sys.path.append('/Users/ducanhnguyen/Documents/mydeepconcolic/src/cw')
sys.path.append('/Users/ducanhnguyen/Documents/mydeepconcolic/src/neuron_statistics')
sys.path.append('/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models')
sys.path.append('/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/rivf')

from src.c_code.pattern_checking import get_activation_function, compare_two_activation_patterns, \
    get_activation_functions, compare_multiple_activation_patterns

from src.model_loader import initialize_dnn_model_from_name
from os import listdir
from src.utils import utilities


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def read_image(fullpath):
    content = []
    with open(fullpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            content.append(np.asarray(row).astype(int))
    content = np.asarray(content)
    return content


# MODEL_OBJECT = initialize_dnn_model_from_name(sys.argv[1])
# CSV_FOLDER = sys.argv[2]
# N_CLASSES = int(sys.argv[3])
# N_FEATURES = int(sys.argv[4])

if __name__ == '__main__':
    '''
    CONFIG - BEGIN
    '''
    paper = 'z3'  # z3, smtInterpol
    name_models = [
        'mnist_simple', 'mnist_ann_keras', 'mnist_deepcheck', 'mnist_simard',
        #
        'fashionmnist_simple', 'fashionmnist_ann_keras', 'fashionmnist_deepcheck', 'fashionmnist_simard',
        #
        'az_simple', 'az_ann_keras', 'az_deepcheck', 'az_simard'
    ]

    if paper == 'proposal':
        Ks = [  # sensitities
            '0.0', '-5.0', '-10.0', '-20.0', '-50.0'
        ]
    elif paper == 'smtInterpol' or paper == 'z3':
        Ks = [
            1,
            20,
            40
        ]

    ###
    NSEEDs_MAX = 1000000
    '''
    CONFIG - END
    '''

    for m in name_models:
        if str(m).startswith("mnist"):
            N_CLASSES = 10
            N_FEATURES = 784
            SHAPE = 28, 28
        elif str(m).startswith("fashion"):
            N_CLASSES = 10
            N_FEATURES = 784
            SHAPE = 28, 28
        elif str(m).startswith("az"):
            N_CLASSES = 26
            N_FEATURES = 784
            SHAPE = 28, 28
        else:
            continue

        MODEL_OBJECT = initialize_dnn_model_from_name(m)
        model = MODEL_OBJECT.get_model()

        truelabel = MODEL_OBJECT.get_ytrain()
        print(model.summary())

        for K in Ks:
            if paper == 'proposal':
                CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{m}/out(K={K},target-1)'
            elif paper == 'z3':
                CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{m}/out_z3based_{K}most'
            elif paper == 'smtInterpol':
                CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{m}/out_smtInterpolbased_{K}most'

            '''
            Create output folder img
            '''
            if not os.path.exists(CSV_FOLDER):
                continue
            out_folder = f"{CSV_FOLDER}/img"  # f"{Path(csv_folder).parent.absolute()}/img"
            if not os.path.exists(out_folder):
                print(f'Initialize {out_folder}')
                os.mkdir(out_folder)

            '''
            Read original image and its adv
            '''
            analyzed_idxes = []
            all_advs_0_255 = []
            all_oris_0_255 = []
            true_labels = []
            for img_idx in range(0, NSEEDs_MAX):
                csv_subfolder = f'{CSV_FOLDER}/{img_idx}'

                if not os.path.exists(csv_subfolder):
                    continue

                ori_img = []
                csvs = find_csv_filenames(csv_subfolder)
                for file in csvs:
                    fullpath = f"{csv_subfolder}/{file}"
                    img = read_image(fullpath)

                    if '_ori' in fullpath:
                        all_oris_0_255.append(img)  # ASSUME THAT THE ORIGINAL IMAGE IS PREDICTED CORRECTLY!
                    elif '_adv' in fullpath:
                        all_advs_0_255.append(img)
                        analyzed_idxes.append(img_idx)

                # append more to the original set to make the two sets have the same length
                for kdx in range(len(all_advs_0_255) - len(all_oris_0_255)):
                    all_oris_0_255.append(all_oris_0_255[-1])

            if len(analyzed_idxes) == 0:
                # there is no generated image
                continue
            analyzed_idxes = np.asarray(analyzed_idxes).reshape(-1)

            all_oris_0_255 = np.asarray(all_oris_0_255).reshape(-1, N_FEATURES)
            pred_ori_label = np.argmax(model.predict(all_oris_0_255 / 255), axis=1)

            all_advs_0_255 = np.asarray(all_advs_0_255).reshape(-1, N_FEATURES)
            pred_adv_label = np.argmax(model.predict(all_advs_0_255 / 255), axis=1)

            print(f'all_oris.shape = {all_oris_0_255.shape}')
            print(f'all_advs.shape = {all_advs_0_255.shape}')
            print(f'pred_adv_label = {pred_adv_label}')
            print(f'pred_ori_label = {pred_ori_label}')

            if len(analyzed_idxes) == 0:
                continue

            '''
            Find out corrected predicted original images
            '''
            zzz = []
            for idx in range(0, len(analyzed_idxes)):
                print(f'seed {analyzed_idxes[idx]}')
                if pred_ori_label[idx] != pred_adv_label[idx] and \
                        pred_ori_label[idx] == truelabel[analyzed_idxes[idx]]:
                    zzz.append(idx)
                    print('satisfy')
                else:
                    print('fail')

            if len(zzz) == 0:
                continue
            zzz = np.asarray(zzz)
            all_oris_0_255 = all_oris_0_255[zzz]
            all_advs_0_255 = all_advs_0_255[zzz]
            pred_adv_label = pred_adv_label[zzz]
            pred_ori_label = pred_ori_label[zzz]
            analyzed_idxes = analyzed_idxes[zzz]

            l0s = utilities.compute_l0s(all_oris_0_255, all_advs_0_255, n_features=N_FEATURES, scale_0_255=True)
            l2s = utilities.compute_l2s(all_oris_0_255, all_advs_0_255, n_features=N_FEATURES)

            ap1 = get_activation_functions(model, all_oris_0_255 / 255)
            ap2 = get_activation_functions(model, all_advs_0_255 / 255)
            is_same_ap = compare_multiple_activation_patterns(ap1, ap2)

            """
            EXPORT TO FILE
            """
            MAX_IMAGES = 10
            with open(f"{out_folder}/advs.csv", mode='w') as f:
                seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                seed.writerow(['index', 'ori label', 'adv label', 'l0', 'l2', 'same pattern'])
                for selected_idx in range(0, len(zzz)):
                    # print(f'Analyze {selected_idx}')

                    newrow = [analyzed_idxes[selected_idx], int(pred_ori_label[selected_idx]),
                              int(pred_adv_label[selected_idx]), l0s[selected_idx], np.round(l2s[selected_idx], 4)]
                    if is_same_ap[selected_idx]:
                        newrow.append('True')
                    else:
                        newrow.append('False')

                    #
                    for feature in all_advs_0_255[selected_idx]:
                        newrow.append(feature)

                    newrow = np.asarray(newrow)
                    seed.writerow(newrow)

                    if MAX_IMAGES > 0:
                        utilities.show_two_images(x_28_28_left=all_oris_0_255[selected_idx].reshape(SHAPE),
                                                  left_title=f'ori (label {int(pred_ori_label[selected_idx])})',
                                                  right_title=f'adv (label {int(pred_adv_label[selected_idx])})',
                                                  x_28_28_right=all_advs_0_255[selected_idx].reshape(SHAPE),
                                                  path=f'{out_folder}/somesamples_idx{analyzed_idxes[selected_idx]}.png',
                                                  display=False)
                        MAX_IMAGES -= 1
