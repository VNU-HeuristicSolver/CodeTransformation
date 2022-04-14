import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from src.c_code.exp.load import get_Xtrain_az_dataset, get_correct_seeds
from src.config_parser import get_config
from src.saved_models.az.az_deepcheck.az_deepcheck import AZ_DEEPCHECK
from src.utils import utilities

if __name__ == '__main__':
    Ks = ['0.0', '-5.0', '-10.0', '-20.0', '-50.0']
    # Ks = ['0.0']
    distributions = dict()
    for K in Ks:
        distributions[K] = []

    N_MAX_L0 = 15
    x_axis = np.arange(1, N_MAX_L0 + 1)

    # models = ['mnist_simple', 'mnist_ann_keras', 'mnist_deepcheck', 'mnist_simard']
    # models = ['fashionmnist_simple', 'fashionmnist_ann_keras', 'fashionmnist_deepcheck', 'fashionmnist_simard']

    models = ['az_ann_keras', 'az_ann_keras', 'az_deepcheck', 'az_simard']
    dataset = get_Xtrain_az_dataset()

    nvalid_ori = get_correct_seeds()

    for model in models:
        ori_by_l0_all = []
        for nl0 in range(0, N_MAX_L0 + 1):
            ori_by_l0_all.append(0)

        for K in Ks:
            print(f'-------------\nMODEL {model}, K = {K}')
            name_plot = f'{model} + K={K}'
            CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{model}/out(K={K},target-1)/img'
            CSV_FILE = f'{CSV_FOLDER}/advs.csv'
            IMAGES = f'{CSV_FOLDER}/some advs'
            if not os.path.exists(IMAGES):
                os.remove(IMAGES)
                os.mkdir(IMAGES)

            if not os.path.exists(CSV_FILE):
                continue
            # read data
            rows = []
            with open(CSV_FILE) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    rows.append(row)

            rows = np.asarray(rows[1:])  # ignore header
            INDEX = 0

            ORILABEL = 1
            ADVLABEL = 2
            L0 = 3
            L2 = 4
            SAME_PATTERN = 5

            samePat_1pixel = 0
            diffPat_1pixel = 0

            samePat = 0
            diffPat = 0
            total_adv = 0

            nSuccessedOri_allpat_1pixel = set()
            nSuccessedOri_allpat = set()

            nSuccessedOri_diffpat_1pixel = set()
            nSuccessedOri_diffpat = set()

            nSuccessedOri_samepat_1pixel = set()
            nSuccessedOri_samepat = set()

            all_advs_1pixel = 0
            all_advs = 0

            MAX_IMAGES = 100
            for idx, row in enumerate(rows):
                if not os.path.exists(
                        f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{model}/out(K={K},target-1)/{row[INDEX]}'):
                    continue

                if int(row[ORILABEL]) != int(row[ADVLABEL]):
                    total_adv += 1

                if int(row[L0]) == 1 and row[SAME_PATTERN] == 'True':
                    samePat_1pixel += 1
                    all_advs_1pixel += 1
                    nSuccessedOri_allpat_1pixel.add(row[INDEX])
                    nSuccessedOri_samepat_1pixel.add(row[INDEX])

                elif int(row[L0]) == 1 and row[SAME_PATTERN] == 'False':
                    diffPat_1pixel += 1
                    all_advs_1pixel += 1
                    nSuccessedOri_allpat_1pixel.add(row[INDEX])
                    nSuccessedOri_diffpat_1pixel.add(row[INDEX])

                if row[SAME_PATTERN] == 'True':
                    samePat += 1
                    all_advs += 1
                    nSuccessedOri_allpat.add(row[INDEX])
                    nSuccessedOri_samepat.add(row[INDEX])

                elif row[SAME_PATTERN] == 'False':
                    diffPat += 1
                    all_advs += 1
                    nSuccessedOri_allpat.add(row[INDEX])
                    nSuccessedOri_diffpat.add(row[INDEX])


                '''
                Export images
                '''
                MAX_IMAGES -= 1
                if MAX_IMAGES > 0:
                    image = np.asarray(row[6:]).astype(int)
                    image = image / 255.
                    image = image.reshape(28, 28)
                    title = f'L0 = {row[L0]}, L2 = {np.round(float(row[L2]), decimals=2)}'
                    path = f'{IMAGES}/idx{row[INDEX]}_ori{row[ORILABEL]}_adv{row[ADVLABEL]}.png'
                    utilities.show_image(image, left_title=title,
                                         path=path,
                                         display=False)

            # print(f'#adv = {total_adv}')
            # print(f'\t#adv - SAME = {samePat}')
            # print(f'\t#adv - DIFF = {diffPat}')
            #
            # print(f'\t#successed ori (SAME) = {len(nSuccessedOri_samepat)}')
            # print(f'\t#successed ori (DIFF) = {len(nSuccessedOri_diffpat)}')
            #
            # print(f'\t#successed ori (ALL) = {len(nSuccessedOri_allpat)}')
            # print(f'\t#success rate (ALL) = {np.round(100 * len(nSuccessedOri_allpat) / nvalid_ori[model], 3)}')

            # print()
            # print(f'#adv (l0 = 1) = {all_advs_1pixel}, in which')
            # print(f'\t#adv (l0r = 1) - SAME = {samePat_1pixel}')
            # print(f'\t#adv (l0 = 1) - DIFF = {diffPat_1pixel}')
            # print(f'\t#successed ori in 1-pixel attack (SAME) = {len(nSuccessedOri_samepat_1pixel)}')
            # print(f'\t#successed ori in 1-pixel attack (DIFF) = {len(nSuccessedOri_diffpat_1pixel)}')
            # print(f'\t#successed ori in 1-pixel attack (ALL) = {len(nSuccessedOri_allpat_1pixel)}')
            # print(f'\tsuccess rate = {np.round(100 * len(nSuccessedOri_allpat_1pixel)/nvalid_ori[model], 2)}')

            '''
            Count the number of advs and successfully modified original inputs
            '''
            # print()
            adv_by_l0 = []
            ori_by_l0 = []
            analyzed = set()
            for nl0 in range(0, N_MAX_L0 + 1):
                adv_by_l0.append(0)
                ori_by_l0.append(0)

            for nl0 in range(0, N_MAX_L0 + 1):
                n_adv = 0
                for idx, row in enumerate(rows):
                    if not os.path.exists(
                            f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{model}/out(K={K},target-1)/{row[INDEX]}'):
                        continue
                    if int(row[L0]) == nl0:
                        adv_by_l0[nl0] += 1

                        if int(row[INDEX]) not in analyzed:
                            analyzed.add(int(row[INDEX]))
                            ori_by_l0[nl0] += 1  # number of modified original input images
                            ori_by_l0_all[nl0] += 1

            adv_by_l0 = adv_by_l0[1:]
            ori_by_l0 = ori_by_l0[1:]

            # print(f'# successfully modified original images = {ori_by_l0}')
            # print(f'sum = {np.sum(ori_by_l0)}')

            SR_by_l0 = np.asarray(ori_by_l0) / int(nvalid_ori[model]) * 100.
            distributions[K].append(SR_by_l0)

            # plt.bar(x_axis, adv_by_l0)
            # plt.plot(x_axis, adv_by_l0, label=f'{model}, K = {K}')
            # plt.plot(x_axis, ori_by_l0, label=f'K = {K}')

        print('------------')
        ori_by_l0_all = ori_by_l0_all[1:]
        # print(f'For all K: # successfully modified original images = {ori_by_l0_all}')
        # print(f'sum = {np.sum(ori_by_l0_all)}')

        # sr
        # plt.xticks(x_axis)
        # plt.xlabel('L0')
        # plt.ylabel('Success rate')
        # plt.legend()
        # plt.savefig(f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/SR-{model}.eps', format='eps')
        # plt.close()


    # average sr
    # for key in distributions.keys():
    #     statis = distributions[key]
    #
    #     # print(f'before {statis}')
    #     # for idx in range(1, len(statis)):
    #     #     for jdx in range(2, len(statis[idx])):
    #     #         statis[idx][jdx] += statis[idx][jdx - 1]
    #     # print(f'after {statis}')
    #
    #     statis = np.average(statis, axis=0).reshape(-1)
    #     plt.plot(x_axis, statis, label = key)


    # plt.xticks(x_axis)
    # plt.title("Fashion-MNIST and MNIST")
    # plt.xlabel('L0')
    # plt.ylabel('Average Success rate')
    # plt.legend()
    # plt.savefig(f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/ASR-all.eps', format='eps')
    # plt.show()
    # plt.close()

