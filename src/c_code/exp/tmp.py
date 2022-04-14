'''
Delete incorrectly predicted seeds
'''
from src.model_loader import initialize_dnn_model_from_name
import os
import numpy as np

if __name__ == '__main__':
    N_SEEDS = 500

    # NAME_MODELS = ['mnist_simple', 'mnist_ann_keras', 'mnist_deepcheck', 'mnist_simard']
    NAME_MODELS = ['fashionmnist_simple', 'fashionmnist_ann_keras', 'fashionmnist_deepcheck', 'fashionmnist_simard']
    # NAME_MODELS = ['az_simple', 'az_ann_keras', 'az_deepcheck', 'az_simard']

    Z3 = False
    # Ks = ['0.0', '-5.0', '-10.0', '-20.0', '-50.0']
    Ks = [1, 20, 40]
    out = "["

    for NAME_MODEL in NAME_MODELS:
        model_object = initialize_dnn_model_from_name(NAME_MODEL)
        Xtrain = model_object.get_Xtrain().reshape(-1, 784)
        model = model_object.get_model()

        model.summary()
        pred_label = np.argmax(model.predict(Xtrain), axis=1)
        ytrue = model_object.get_ytrain()

        correct = np.where(ytrue == pred_label)
        correct = np.asarray(correct)

        # for K in Ks:
        #     FOLDER = f"/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{NAME_MODEL}/out(K={K},target-1)"
        #     for idx in range(0, 1000):
        #         seed_folder = f"{FOLDER}/{idx}"
        #         if os.path.exists(seed_folder) and idx not in correct:
        #             out += f"\"{seed_folder}\", "
        #             # os.remove(seed_folder)

        for K in Ks:
            if Z3:
                FOLDER = f"/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{NAME_MODEL}/out_z3based_{K}most"
            else:
                FOLDER = f"/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{NAME_MODEL}/out_smtInterpolbased_{K}most"
            for idx in range(0, 1000):
                seed_folder = f"{FOLDER}/{idx}"
                if os.path.exists(seed_folder) and idx not in correct:
                    out += f"\"{seed_folder}\", "
                    # os.remove(seed_folder)

    out += "]"
    print(out)