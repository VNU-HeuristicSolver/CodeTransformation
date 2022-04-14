import csv
import os

import numpy as np
import tensorflow

from src.c_code.exp.load import get_correct_seeds, load_dataset, get_folder_name
from src.utils import utilities

MODELS = [
    'mnist_ann_keras',
    'fashionmnist_ann_keras',
    'az_ann_keras'

    # 'mnist_simard', 'mnist_ann_keras', 'mnist_simple', 'mnist_deepcheck',
    # 'fashionmnist_simard', 'fashionmnist_ann_keras', 'fashionmnist_simple', 'fashionmnist_deepcheck',
    # 'az_simard', 'az_ann_keras', 'az_simple', 'az_deepcheck'
]

SOLVER = 'proposal'  # smtInterpol, proposal, Z3
if SOLVER == 'smtInterpol' or SOLVER == 'Z3':
    Ks = [
        1
        ,
        20
        ,
        40
    ]
else:
    Ks = [
        # '0.0'
        # ,
        # '-5.0'
        # ,
        # '-10.0'
        # ,
        # '-20.0'
        # ,
        '-50.0'
    ]

INDEX, ORILABEL, ADVLABEL, L0, L2, SAME_PATTERN, START_ADV = 0, 1, 2, 3, 4, 5, 6  # depends on the structures of advs

if __name__ == '__main__':
    nvalid_ori = get_correct_seeds()
    for model in MODELS:
        DATASET, shape = load_dataset(model)
        ssim_graph = dict()
        for idx in range(0, 100):
            ssim_graph[idx] = []

        for K in Ks:
            print(f'-------------\nMODEL {model}, K = {K}')

            CSV_FOLDER = get_folder_name(SOLVER, model, K)
            if not os.path.exists(CSV_FOLDER):
                continue

            CSV_FILE = f'{CSV_FOLDER}/img/advs.csv'
            if not os.path.exists(CSV_FILE):
                continue

            IMAGES = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/outL10below/{model}'
            if not os.path.exists(IMAGES):
                os.mkdir(IMAGES)

            # read data in csv file
            rows = []
            with open(CSV_FILE) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    rows.append(row)

            rows = np.asarray(rows[1:])  # ignore header

            for idx, row in enumerate(rows):
                if not os.path.exists(f'{CSV_FOLDER}/{row[INDEX]}'):
                    print('Ignore')
                    continue
                '''
                Export images
                '''
                adv = np.asarray(row[START_ADV:]).astype(int)
                if np.max(adv) > 1:
                    adv = adv / 255.
                adv = adv.reshape(shape)

                ori = DATASET[int(row[INDEX])]
                ori = ori.reshape(shape)
                im1 = tensorflow.image.convert_image_dtype(ori.reshape(1, 28, 28, 1), tensorflow.float32)
                im2 = tensorflow.image.convert_image_dtype(adv.reshape(1, 28, 28, 1), tensorflow.float32)
                ssim = tensorflow.image.ssim(im1, im2, max_val=1.0);
                ssim = ssim.numpy()[0]
                ssim = np.round(ssim, 5);

                ssim_graph[int(row[L0])].append(ssim)

        plot = []
        for idx in range(0, 100):
            plot.append(0);
        for key in ssim_graph.keys():
            # print(key)
            arr = ssim_graph.get(key)
            if arr is not None:
                avg = np.average(arr)
                plot[int(key)] = avg
            else:
                plot[int(key)] = 0

        # plot = np.mean(plot, axis=1)
        print(f'{model}: sism = {plot}')
                # if int(row[L0]) <= 10:

                #
                #     title = f'L0 = {row[L0]}, L2 = {np.round(float(row[L2]), decimals=2)}'
                #
                #     pat = "samePattern"
                #     if row[SAME_PATTERN] == 'False':
                #         pat = 'diffPattern'
                #
                #
                #
                #
                #     path = f'{IMAGES}/K{K}_idx{row[INDEX]}_ori{row[ORILABEL]}_adv{row[ADVLABEL]}_lzero{row[L0]}_{pat}_ssim{ssim}.png'
                #     print(path)
                #
                #
                #     utilities.show_two_images(x_28_28_left=ori,
                #                               left_title=f'label {row[ORILABEL]}',
                #                               right_title=f'label {row[ADVLABEL]}',
                #                               x_28_28_right=adv,
                #                               path=path,
                #                               display=False)
                # utilities.show_image(image, left_title=title,
                #                      path=path,
                #                      display=False)
