import os, glob
from DataGenerator import DataGenerator

base_path = 'E:/processed'

def get_trainval():
    x_train_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
    x_val_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

    train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=16, dim=(44,44), n_channels=3, n_classes=None, shuffle=True)
    val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=16, dim=(44,44), n_channels=3, n_classes=None, shuffle=False)

    return train_gen, val_gen

def get_xytest():
    x_test_list = sorted(glob.glob(os.path.join(base_path, 'x_test', '*.npy')))
    y_test_list = sorted(glob.glob(os.path.join(base_path, 'y_test', '*.npy')))

    return x_test_list, y_test_list


