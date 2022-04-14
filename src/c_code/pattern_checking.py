'''
Given two images, check if there two iamges have the same activation pattern.
'''
from tensorflow.python.keras import Model

from src.model_loader import initialize_dnn_model_from_name
from src.utils import keras_layer, keras_activation
import os
import csv
import numpy as np


def get_name_neuron(layer, idx):
    return f"l{layer}_{idx}"


def get_activation_functions(model, images_0_1):  # for multiple image
    images_0_1 = images_0_1.reshape(-1, 784)
    activation_pattern = []
    for kdx in range(len(images_0_1)):
        activation_pattern.append(dict())
    activation_pattern = np.asarray(activation_pattern)

    before_softmax = -2

    # iterate atmoic layers
    for layer_idx, layer in enumerate(model.layers[:before_softmax]):  # ignore presoftmax layer

        if keras_layer.is_dense(layer):
            intermediate_layer_model = Model(inputs=model.inputs,
                                             outputs=layer.output)
            neurons = intermediate_layer_model.predict(images_0_1)
            num_neuron = keras_layer.get_number_of_units(model, layer_idx)

            neuron_idx = 0
            for jdx in range(num_neuron):
                neuron_idx += 1

                # save name of neuron
                name_neuron = get_name_neuron(layer_idx, jdx)
                # compute score
                for kdx in range(len(images_0_1)):
                    neuron = neurons[kdx][jdx]
                    if neuron >= 0:
                        activation_pattern[kdx][name_neuron] = True
                    else:
                        activation_pattern[kdx][name_neuron] = False

    return activation_pattern


def get_activation_function(model, image_0_1):
    image_0_1 = image_0_1.reshape(-1, 784)
    activation_pattern = dict()
    before_softmax = -2

    # iterate atmoic layers
    for layer_idx, layer in enumerate(model.layers[:before_softmax]):  # ignore presoftmax layer

        if keras_layer.is_dense(layer):
            intermediate_layer_model = Model(inputs=model.inputs,
                                             outputs=layer.output)
            neurons = intermediate_layer_model.predict(image_0_1)
            num_neuron = keras_layer.get_number_of_units(model, layer_idx)

            neuron_idx = 0
            for jdx in range(num_neuron):
                neuron_idx += 1

                # save name of neuron
                name_neuron = get_name_neuron(layer_idx, jdx)
                # compute score
                neuron = neurons[0][jdx]
                if neuron >= 0:
                    activation_pattern[name_neuron] = True
                else:
                    activation_pattern[name_neuron] = False

    return activation_pattern


def compare_multiple_activation_patterns(ap1, ap2):
    ap = []
    for idx in range(len(ap1)):
        isTheSame = True

        if len(ap1[idx]) != len(ap2[idx]):
            isTheSame = False
        else:
            for k1 in ap1[idx].keys():
                v1 = ap1[idx].get(k1)
                v2 = ap2[idx].get(k1)
                if v1 != v2:
                    isTheSame = False
                    break

        ap.append(isTheSame)
    return ap


def compare_two_activation_patterns(ap1: dict(), ap2: dict()):
    if len(ap1) != len(ap2):
        return False

    for k1 in ap1.keys():
        v1 = ap1.get(k1)
        v2 = ap2.get(k1)
        if v1 != v2:
            return False
        # else:
        #     print('Pass')

    return True


def get_image(index, folder, Xtrain):
    adv_path = f"{folder}/{index}/{index}_adv.csv"
    ori = None # [0..255]
    adv = None # [0..255]
    if os.path.exists(adv_path):
        # load adv
        adv = []
        with open(adv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                adv.append(np.asarray(row).astype(int))
        adv = np.asarray(adv).reshape(-1)
        #
        ori = Xtrain[index].reshape(-1)

    return ori, adv


if __name__ == '__main__':
    N_SEEDS = 500
    NAME_MODEL = "mnist_simple"
    FOLDER = "/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/mnist_simple/out_z3based"

    model_object = initialize_dnn_model_from_name(NAME_MODEL)
    Xtrain = model_object.get_Xtrain().reshape(-1, 784)
    model = model_object.get_model()
    model.summary()

    same_activation_patterns = []
    for idx in range(N_SEEDS):
        ori_0_255, adv_0_255 = get_image(idx, FOLDER, Xtrain)
        ori_0_1 = ori_0_255 / 255
        adv_0_1 = adv_0_255 / 255
        if ori_0_1 is not None and adv_0_1 is not None:
            print(f'\nAnalyze {idx}')

            ori_pred = model.predict(ori_0_1.reshape(-1, 784))
            # print(f"Ori pred = {ori_pred}")
            adv_pred = model.predict(adv_0_1.reshape(-1, 784))
            # print(f"Adv pred = {adv_pred}")

            if np.argmax(ori_pred, axis=1) != np.argmax(adv_pred, axis=1):
                ap1 = get_activation_function(model, ori_0_1)
                # print(f"Activation {ap1}")
                ap2 = get_activation_function(model, adv_0_1)
                # print(f"Activation {ap2}")

                is_same_ap = compare_two_activation_patterns(ap1, ap2)
                if is_same_ap:
                    same_activation_patterns.append(idx)

    print(f"same_activation_patterns = {same_activation_patterns}")
    print(f"size = {len(same_activation_patterns)}")
