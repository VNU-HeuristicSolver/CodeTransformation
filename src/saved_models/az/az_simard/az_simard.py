from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

from src.PATH import SRC_ROOT, DATASET_ROOT
from src.config_parser import get_config
from src.saved_models.az.az_dataset import az_dataset
import numpy as np

'''
Overall training score: 0.0023838207125663757
Accuracy on train set: 0.9992817640304565
Overall test score: 0.11899544298648834
Accuracy on test set: 0.9902536273002625

'''


class AZ_SIMARD(az_dataset):

    def __init__(self):
        super(AZ_SIMARD, self).__init__()

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
    # train model
    model = 'LOAD'

    # train model
    if model == 'TRAIN':
        az = AZ_SIMARD()
        az.set_num_classes(26)

        az.train_model(train=True,
                       kernel_path=f'{SRC_ROOT}/saved_models/az/az_simard/az_simard.h5',
                       model_path=f'{SRC_ROOT}/saved_models/az/az_simard/az_simard.json',
                       training_path=f'{DATASET_ROOT}/a-z handwritten/training.csv',
                       testing_path=f'{DATASET_ROOT}/a-z handwritten/test.csv',
                       nb_epoch=200)

    if model == 'LOAD':
        # load model
        name_model = "az_simard"
        model_object = AZ_SIMARD()
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

        numseed = 500
        pred_label = np.argmax(model.predict(model_object.get_Xtrain()[:numseed]), axis=1)
        correct_predictions = np.sum(pred_label == model_object.get_ytrain()[:numseed])
        training_acc = correct_predictions / len(model_object.get_ytrain()[:numseed])
        print(f'training_acc = {training_acc}')
        print(f'correct_predictions = {correct_predictions}')

        pred_label = np.argmax(model.predict(model_object.get_Xtest()), axis=1)
        test_acc = np.sum(pred_label == model_object.get_ytest()) / len(model_object.get_ytest())
        print(f'test_acc = {test_acc}')

        #
        # plot an observation
        import matplotlib.pyplot as plt
        x_train, y_train = model_object.get_an_observation(index=100)
        img = x_train.reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f'A sample')
        plt.show()

# training_acc = 0.9842394423393822
# correct_predictions = 293263
# test_acc = 0.975459463813449