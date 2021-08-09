import MNN
import os
import random
from cores.dataLoader import DataLoader

class DataRead(object):
    def __init__(self,
                 data_dir=None,
                 cache_dir=None,
                 batch_size=32,
                 mean_value=127.5,
                 scale=1./127.5):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.mean_value = mean_value
        self.scale = scale
        self.train_list = list()
        self.val_list = list()

    def get_class_index_dict(self):
        class_index_dict = {}
        class_names = os.listdir(self.data_dir)
        index = 0
        for name in class_names:
            if os.path.isdir(os.path.join(self.data_dir, name)):
                class_index_dict[name] = index
                index += 1
        return class_index_dict

    def get_train_val_loader(self, train_split_ratio):
        self._split_train_val_dataset(train_split_ratio)
        train_loader = DataLoader(data_list=self.train_list,
                                  batch_size=self.batch_size,
                                  is_training=True,
                                  cache_path=self.cache_dir,
                                  mean_value=self.mean_value,
                                  scale=self.scale)

        val_loader = DataLoader(data_list=self.val_list,
                                batch_size=self.batch_size,
                                is_training=False,
                                cache_path=self.cache_dir,
                                mean_value=self.mean_value,
                                scale=self.scale)

        return train_loader,val_loader

    def _split_train_val_dataset(self, train_split_ratio):
        class_names = os.listdir(self.data_dir)
        class_index_dict = self.get_class_index_dict()
        SEED = 1
        for name in class_names:
            total_image_list = list()
            class_dir = os.path.join(self.data_dir, name)
            if not os.path.isdir(class_dir):
                print(class_dir)
                continue
            images = os.listdir(class_dir)
            for image in images:
                if 'jpg' in image or 'png' in image or 'jpeg' in image:
                    total_image_list.append({os.path.join(class_dir, image): class_index_dict[name]})
            cur_class_total_number = len(total_image_list)
            train_example_number = int(cur_class_total_number * train_split_ratio)
            #@TODO:check open seed could make train better
            #randome.seed(SEED)
            random.shuffle(total_image_list)
            for i in range(train_example_number):
                self.train_list.append(total_image_list[i])
            for i in range(train_example_number, cur_class_total_number):
                self.val_list.append(total_image_list[i])
                    #if random.random() < train_split_ratio:
                    #    self.train_list.append({os.path.join(class_dir, image): class_index_dict[name]})
                    #else:
                    #    self.val_list.append({os.path.join(class_dir, image): class_index_dict[name]})


