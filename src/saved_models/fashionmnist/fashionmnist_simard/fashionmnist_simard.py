from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

from src.PATH import DATASET_ROOT, SRC_ROOT
from src.config_parser import get_config
from src.saved_models.fashionmnist.fashionmnist_dataset import fashionmnist_dataset
import numpy as np

import os
'''
Overall training score: 0.012811495922505856
Accuracy on train set: 0.9957833290100098
Overall test score: 0.9800440669059753
Accuracy on test set: 0.8952000141143799
'''
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class FASHIONMNIST_SIMARD(fashionmnist_dataset):

    def __init__(self):
        super(FASHIONMNIST_SIMARD, self).__init__()

    def create_model(self, input_shape):
        model = Sequential()

        model.add(Dense(128, name='dense_1', input_dim=input_shape))
        model.add(Activation('relu', name='relu_1'))

        model.add(Dense(64, name='dense_2'))
        model.add(Activation('relu', name='relu_2'))

        model.add(Dense(32, name='dense_3'))
        model.add(Activation('relu', name='relu_3'))

        model.add(Dense(16, name='dense_4'))
        model.add(Activation('relu', name='relu_4'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model


if __name__ == '__main__':
    model = 'LOAD'

    # train model
    if model == 'TRAIN':
        mnist = FASHIONMNIST_SIMARD()
        mnist.set_num_classes(10)

        mnist.train_model(train=True,

                          kernel_path=f'{SRC_ROOT}/saved_models/fashionmnist/fashionmnist_simard/fashionmnist_simard.h5',
                          model_path=f'{SRC_ROOT}/saved_models/fashionmnist/fashionmnist_simard/fashionmnist_simard.json',
                          training_path=f'{DATASET_ROOT}/fashion-mnist/fashion-mnist_train.csv',
                          testing_path=f'{DATASET_ROOT}/fashion-mnist/fashion-mnist_test.csv',

                          nb_epoch=300)

    if model == 'LOAD':
        # load model
        name_model = "fashionmnist_simard"
        model_object = FASHIONMNIST_SIMARD()
        model_object.set_num_classes(get_config([name_model, "num_classes"]))
        model_object.read_data(trainset_path=get_config([name_model, "train_set"]),
                               testset_path=get_config([name_model, "test_set"]))
        model_object.load_model(weight_path=get_config([name_model, "weight"]),
                                structure_path=get_config([name_model, "structure"]),
                                trainset_path=get_config([name_model, "train_set"]))
        model_object.set_name_dataset(name_model)
        model_object.set_image_shape((28, 28))
        model_object.set_selected_seed_index_file_path(get_config(["files", "selected_seed_index_file_path"]))
        model_object.get_model().summary()

        model = model_object.get_model()

        numseed = None
        pred_label = np.argmax(model.predict(model_object.get_Xtrain()[:numseed]), axis=1)
        correct_predictions = np.sum(pred_label == model_object.get_ytrain()[:numseed])
        training_acc = correct_predictions / len(model_object.get_ytrain()[:numseed])
        print(f'training_acc = {training_acc}')
        print(f'correct_predictions = {correct_predictions}')

        pred_label = np.argmax(model.predict(model_object.get_Xtest()), axis=1)
        test_acc = np.sum(pred_label == model_object.get_ytest()) / len(model_object.get_ytest())
        print(f'test_acc = {test_acc}')

        #
        # # plot an observation
        # import matplotlib.pyplot as plt
        # x_train, y_train = mnist.get_an_observation(index=516)
        # img = x_train.reshape(28, 28)
        # plt.imshow(img, cmap='gray')
        # plt.title(f'A sample')
        # plt.show()


