import csv
import os

import numpy as np

from src.c_code.exp.load import get_correct_seeds, load_dataset, get_folder_name
from src.utils import utilities

MODELS = [
    'mnist_simard', 'mnist_ann_keras', 'mnist_simple', 'mnist_deepcheck',
    'fashionmnist_simard', 'fashionmnist_ann_keras', 'fashionmnist_simple', 'fashionmnist_deepcheck',
    'az_simard', 'az_ann_keras', 'az_simple', 'az_deepcheck']

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
          '-5.0'
        ,
          '-10.0'
        ,
          '-20.0'
        ,
          '-50.0'
    ]

INDEX, ORILABEL, ADVLABEL, L0, L2, SAME_PATTERN, START_ADV = 0, 1, 2, 3, 4, 5, 6  # depends on the structures of advs

if __name__ == '__main__':
    nvalid_ori = get_correct_seeds()
    for model in MODELS:
        DATASET, shape = load_dataset(model)
        for K in Ks:
            print(f'-------------\nMODEL {model}, K = {K}')
            name_plot = f'{model} + K={K}'

            CSV_FOLDER = get_folder_name(SOLVER, model, K)
            if not os.path.exists(CSV_FOLDER):
                continue

            CSV_FILE = f'{CSV_FOLDER}/img/advs.csv'
            if not os.path.exists(CSV_FILE):
                continue

            IMAGES = f'{CSV_FOLDER}/img/some images'
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

            MAX_RENDER_IMAGES = 100000
            for idx, row in enumerate(rows):
                if not os.path.exists(f'{CSV_FOLDER}/{row[INDEX]}'):
                    print('Ignore')
                    continue
                '''
                Export images
                '''
                MAX_RENDER_IMAGES -= 1
                if MAX_RENDER_IMAGES > 0:

                    adv = np.asarray(row[START_ADV:]).astype(int)
                    if np.max(adv) > 1:
                        adv = adv / 255.
                    adv = adv.reshape(shape)

                    ori = DATASET[int(row[INDEX])]
                    ori = ori.reshape(shape)

                    title = f'L0 = {row[L0]}, L2 = {np.round(float(row[L2]), decimals=2)}'

                    pat = "samePattern"
                    if row[SAME_PATTERN] == 'False':
                        pat = 'diffPattern'
                    path = f'{IMAGES}/idx{row[INDEX]}_ori{row[ORILABEL]}_adv{row[ADVLABEL]}_lzero{row[L0]}_{pat}.png'

                    print(path)
                    utilities.show_two_images(x_28_28_left=ori,
                                              left_title=f'label {row[ORILABEL]}',
                                              right_title=f'label {row[ADVLABEL]}',
                                              x_28_28_right=adv,
                                              path=path,
                                              display=False)
                    # utilities.show_image(image, left_title=title,
                    #                      path=path,
                    #                      display=False)
