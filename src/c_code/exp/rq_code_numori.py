import csv
import os

import numpy as np

from src.c_code.exp.load import get_correct_seeds

MODELS = [
    'mnist_simard', 'mnist_ann_keras', 'mnist_simple', 'mnist_deepcheck',
    'fashionmnist_simard', 'fashionmnist_ann_keras', 'fashionmnist_simple', 'fashionmnist_deepcheck',
    'az_simard', 'az_ann_keras', 'az_simple', 'az_deepcheck']

SOLVER = 'proposal'  # smtInterpol, proposal, Z3
if SOLVER == 'smtInterpol' or SOLVER == 'Z3':
    Ks = [
        1
        # ,
        # 20
        # ,
        # 40
    ]
else:
    Ks = [
        '0.0'
        # ,
        #   '-5.0'
        # ,
        #   '-10.0'
        # ,
        #   '-20.0'
        # ,
        #   '-50.0'
    ]

if __name__ == '__main__':
    xxx = False
    yyy = False
    zzz = False
    kkk = True
    distributions = dict()
    for K in Ks:
        distributions[K] = []

    N_MAX_L0 = 15
    x_axis = np.arange(1, N_MAX_L0 + 1)

    nvalid_ori = get_correct_seeds()

    for model in MODELS:

        ori_by_l0_all = []
        for nl0 in range(0, N_MAX_L0 + 1):
            ori_by_l0_all.append(0)

        for K in Ks:
            # print(f'-------------\nMODEL {model}, K = {K}')
            name_plot = f'{model} + K={K}'

            if SOLVER == 'Z3':
                CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{model}/out_z3based_{K}most'
            elif SOLVER == 'smtInterpol':
                CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{model}/out_smtInterpolbased_{K}most'
            else:
                CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{model}/out(K={K},target-1)'

            # print(f'CSV_FOLDER = {CSV_FOLDER}')
            CSV_FILE = f'{CSV_FOLDER}/img/advs.csv'

            if not os.path.exists(CSV_FILE):
                if yyy:
                    print(f'{model}, {0.0}')
                if kkk:
                    print(f'{model}, {0.0}%')

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
                        f'{CSV_FOLDER}/{row[INDEX]}'):
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

            if zzz:
                # nSuccessedOri_diffpat_1pixel = np.asarray(nSuccessedOri_diffpat_1pixel)
                # nSuccessedOri_samepat_1pixel = np.asarray(nSuccessedOri_samepat_1pixel)
                print(f'{model}, {len(nSuccessedOri_diffpat_1pixel)}, {len(nSuccessedOri_samepat_1pixel)}')

            # print(f'#adv = {total_adv}')
            # print(f'\t#adv - SAME = {samePat}')
            # print(f'\t#adv - DIFF = {diffPat}')
            #
            # print(f'\t#successed ori (SAME) = {len(nSuccessedOri_samepat)}')
            # print(f'\t#successed ori (DIFF) = {len(nSuccessedOri_diffpat)}')

            if kkk:
                # print(f'\t#successed ori (ALL) = {len(nSuccessedOri_allpat)}')
                # print(f'{model}, {len(nSuccessedOri_allpat)}')

                # print(f'\t#success rate (ALL) = {np.round(100 * len(nSuccessedOri_allpat) / nvalid_ori[model], 3)}')
                print(f'{model}, {np.round(100 * len(nSuccessedOri_allpat) / nvalid_ori[model], 2)}%')

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
                            f'{CSV_FOLDER}/{row[INDEX]}'):
                        continue
                    if int(row[L0]) == nl0:
                        adv_by_l0[nl0] += 1

                        if int(row[INDEX]) not in analyzed:
                            analyzed.add(int(row[INDEX]))
                            ori_by_l0[nl0] += 1  # number of modified original input images
                            ori_by_l0_all[nl0] += 1

            adv_by_l0 = adv_by_l0[1:]
            ori_by_l0 = ori_by_l0[1:]

            if xxx:
                print(f'{model}, {ori_by_l0}')

            if yyy:
                SR_by_l0 = np.asarray(ori_by_l0) / int(nvalid_ori[model]) * 100.
                distributions[K].append(SR_by_l0)
                print(f'{model}, {np.round(SR_by_l0[0], 2)}')
            # print(f'# successfully modified original images = {ori_by_l0}')
            # print(f'sum = {np.sum(ori_by_l0)}')

            # plt.bar(x_axis, adv_by_l0)
            # plt.plot(x_axis, adv_by_l0, label=f'{model}, K = {K}')
            # plt.plot(x_axis, ori_by_l0, label=f'K = {K}')

        # print('------------')
        # ori_by_l0_all = ori_by_l0_all[1:]
        # print(f'For all K: # successfully modified original images = {ori_by_l0_all}')
        # print(f'sum = {np.sum(ori_by_l0_all)}')
