from src.config_parser import get_config
from src.saved_models.az.az_deepcheck.az_deepcheck import AZ_DEEPCHECK
from src.saved_models.fashionmnist.fashionmnist_simple.fashionmnist_simple import FASHIONMNIST_SIMPLE
from src.saved_models.mnist.mnist_simple.mnist_simple import MNIST_SIMPLE


def get_Xtrain_az_dataset():
    name_model = "az_deepcheck"
    model_object = AZ_DEEPCHECK()
    model_object.set_num_classes(get_config([name_model, "num_classes"]))
    model_object.read_data(trainset_path=get_config([name_model, "train_set"]),
                           testset_path=get_config([name_model, "test_set"]))
    return model_object.get_Xtrain()

def get_Xtrain_mnist_dataset():
    name_model = "mnist_simple"
    model_object = MNIST_SIMPLE()
    model_object.set_num_classes(get_config([name_model, "num_classes"]))
    model_object.read_data(trainset_path=get_config([name_model, "train_set"]),
                           testset_path=get_config([name_model, "test_set"]))
    return model_object.get_Xtrain()

def get_Xtrain_fashionmnist_dataset():
    name_model = "fashionmnist_simple"
    model_object = FASHIONMNIST_SIMPLE()
    model_object.set_num_classes(get_config([name_model, "num_classes"]))
    model_object.read_data(trainset_path=get_config([name_model, "train_set"]),
                           testset_path=get_config([name_model, "test_set"]))
    return model_object.get_Xtrain()

def get_correct_seeds():
    nvalid_ori = dict()
    nvalid_ori['mnist_simple'] = 491
    nvalid_ori['mnist_ann_keras'] = 499
    nvalid_ori['mnist_deepcheck'] = 482
    nvalid_ori['mnist_simard'] = 500

    nvalid_ori['fashionmnist_simple'] = 460
    nvalid_ori['fashionmnist_ann_keras'] = 483
    nvalid_ori['fashionmnist_deepcheck'] = 442
    nvalid_ori['fashionmnist_simard'] = 496

    nvalid_ori['az_simple'] = 447
    nvalid_ori['az_ann_keras'] = 472
    nvalid_ori['az_deepcheck'] = 434
    nvalid_ori['az_simard'] = 490
    return nvalid_ori

def load_dataset(model):
    if model.startswith("mnist"):
        shape = 28, 28
        return get_Xtrain_mnist_dataset(), shape
    elif model.startswith("fashion"):
        shape = 28, 28
        return get_Xtrain_fashionmnist_dataset(), shape
    elif model.startswith("az"):
        shape = 28, 28
        return get_Xtrain_az_dataset(), shape
    return None

def get_folder_name(method, model, K):
    if method == 'Z3':
        CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{model}/out_z3based_{K}most'
    elif method == 'smtInterpol':
        CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{model}/out_smtInterpolbased_{K}most/'
    else:
        CSV_FOLDER = f'/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/data/proposal/{model}/out(K={K},target-1)'
    return CSV_FOLDER