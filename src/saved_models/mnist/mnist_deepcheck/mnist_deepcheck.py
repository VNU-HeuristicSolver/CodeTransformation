from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

from src.PATH import SRC_ROOT, DATASET_ROOT
from src.config_parser import get_config
from src.saved_models.mnist.mnist_dataset import mnist_dataset
import numpy as np

'''
Overall training score: 0.10900241881608963
Accuracy on train set: 0.9676166772842407
Overall test score: 0.22557860612869263
Accuracy on test set: 0.9440000057220459
'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
class MNIST_DEEPCHECK(mnist_dataset):

    def __init__(self):
        super(MNIST_DEEPCHECK, self).__init__()

    def create_model(self, input_shape):
        model = Sequential()

        model.add(Dense(10, name='dense_1', input_dim=input_shape))
        model.add(Activation('relu', name='relu_1'))

        model.add(Dense(10, name='dense_2'))
        model.add(Activation('relu', name='relu_2'))

        model.add(Dense(10, name='dense_3'))
        model.add(Activation('relu', name='relu_3'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model


if __name__ == '__main__':
    model = 'LOAD'
    if model == 'TRAIN':
        # train model
        mnist = MNIST_DEEPCHECK()
        mnist.set_num_classes(10)

        mnist.train_model(train=True,
                          kernel_path=f'{SRC_ROOT}/saved_models/mnist/mnist_deepcheck/mnist_deepcheck.h5',
                          model_path=f'{SRC_ROOT}/saved_models/mnist/mnist_deepcheck/mnist_deepcheck.json',
                          training_path=f'{DATASET_ROOT}/digit-recognizer/train.csv',
                          testing_path=f'{DATASET_ROOT}/digit-recognizer/test.csv',
                          nb_epoch=100)

    if model == 'LOAD':
        # load model
        name_model = "mnist_deepcheck"
        model_object = MNIST_DEEPCHECK()
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


