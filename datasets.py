import os
import albumentations as A

import CONST
from transforms import load_transform
from dataset.DopplerNetDS import DopplerNetDS


class datas(object):

    def __init__(self, loader_func, dataset_config, input_transform,
                 train_filenames_list: str, val_filenames_list: str, test_filenames_list: str):

        self.loader_func = loader_func
        self.input_transform = input_transform

        self.dataset_config = dataset_config

        self.dataset_config["kpts_info"] = self.create_kpts_info(num_kpts=dataset_config["num_kpts"], closed_contour=dataset_config["closed_contour"])

        assert os.path.exists(dataset_config["img_folder"]), "image repository does not exist."
        print("\n\nCreating trainset...")
        self.trainset = self.load_train(train_filenames_list)
        print("\n\nCreating valset...")
        self.valset = self.load_val(val_filenames_list)
        print("\n\nCreating testset...")
        self.testset = self.load_test(test_filenames_list)

    def create_kpts_info(self, num_kpts: int, closed_contour: bool):
        kpts_info = {'names':[], 'connections':[], 'colors':[]}
        kpts_info['names'] = {}
        for kpt_indx in range(num_kpts):
            kpts_info['names']["kp{}".format(kpt_indx+1)] = kpt_indx
        kpts_info['connections'] = [[i, i+1] for i in range(len(kpts_info['names'])-1)]
        kpts_info['colors'] = [[0, 0, 255], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                               [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
                               [170, 55, 0], [85, 55, 0], [0, 55, 0], [0, 55, 85]
                               ]  # Note: Limited to 12 classes.
        kpts_info['closed_contour'] = closed_contour
        return kpts_info

    def load_train(self, train_filenames_list: str):
        trainset = None
        if train_filenames_list is not None:
            trainset = self.loader_func(dataset_config=self.dataset_config, filenames_list=train_filenames_list, transform=self.input_transform)
        return trainset

    def load_val(self, val_filenames_list: str):
        valset = None
        if val_filenames_list is not None:
            valset = self.loader_func(dataset_config=self.dataset_config, filenames_list=val_filenames_list, transform=None)
        return valset

    def load_test(self, test_filenames_list: str):
        testset = None
        if test_filenames_list is not None:
            testset = self.loader_func(dataset_config=self.dataset_config, filenames_list=test_filenames_list, transform=None)
        return testset



def load_dataset(ds_name: str, num_classes: int, num_kpts: int, allowed_kpts = list[str],
                 input_transform: A.core.composition.Compose = None, input_size: int = 256, num_frames: int = 1) -> datas:

    us_data_folder = CONST.US_MultiviewData

    ## SPECTRAL DOPPLER
    loader_func = DopplerNetDS
    img_dirname = os.path.join(us_data_folder, "frames/")
    anno_dirname = os.path.join(us_data_folder, "annotations/")
    train_filenames_list = os.path.join(us_data_folder, 'filenames/doppler_train_filenames.txt')
    val_filenames_list = os.path.join(us_data_folder, 'filenames/doppler_val_filenames.txt')
    test_filenames_list = os.path.join(us_data_folder, 'filenames/doppler_test_filenames.txt')
    frame_selection_mode = None
    closed_contour = False

    dataset_config = {"root_data_folder": us_data_folder, "img_folder": img_dirname, "anno_folder": anno_dirname, "transform": input_transform, "input_size": input_size, "num_classes": num_classes, 
                      "num_kpts": num_kpts, "allowed_kpts": allowed_kpts, "closed_contour": closed_contour, "num_frames": num_frames, "frame_selection_mode": frame_selection_mode}

    ds = datas(loader_func=loader_func, dataset_config=dataset_config, input_transform=input_transform,
               train_filenames_list=train_filenames_list, val_filenames_list=val_filenames_list, test_filenames_list=test_filenames_list)


    if ds.trainset is not None and ds.testset is not None:
            print("loading dataset : {}.. number of train examples is {}, number of val examples is {}, number of test examples is {}."
                  .format(ds_name, len(ds.trainset), len(ds.valset), len(ds.testset)))
    else:
            print('loading empty dataset.')

    return ds



if __name__ == '__main__':
    ds_name = "prueba"#"debug"#"echonet_random"#"echonet_random"#"echonet_cycle"
    augmentation_type = "strong_echo_cycle" #"strongkeep" #"twochkeep" #"strongkeep"


    #input_transform = None
    input_transform = load_transform()

    print_folder=os.path.join("./visu/", ds_name)

    if not os.path.exists(print_folder):
        os.makedirs(print_folder)
    ds = load_dataset(ds_name=ds_name, input_transform=input_transform)
    g = ds.trainset#ds.valset#ds.trainset
    for k in range(1, 3, 1): #len(g)):
        dat = g.get_img_and_kpts(index=k)
        g.plot_item(k, do_augmentation=False, print_folder=os.path.join("./visu/", ds_name))
        g.plot_item(k, do_augmentation=True, print_folder=os.path.join("./visu/", ds_name))

