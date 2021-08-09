# -*- coding: UTF-8 -*-
import MNN
import os
import random
from cores.clsDataset import ClsDataset

F = MNN.expr 
MNN_DATA = MNN.data

class DataLoader(object):
    def __init__(self, 
                 data_list=list(),
                 batch_size=32, 
                 is_training=True,
                 cache_path='',
                 mean_value=127.5,
                 scale=1./127.5):
        '''init process for a data loader.
        Args:
            batch_size: batch size for training or evaluation process
            is_training: whether is in training mode
            train_path: path to save training cache
            val_path: path to save evaluation cache
        '''
        self.is_training = is_training
        self.data_list = data_list
        self.train_path = os.path.join(cache_path, 'train_cache')
        self.val_path = os.path.join(cache_path, 'val_cache')
        self.data_path = self.train_path if is_training else self.val_path
        self.batch_size = batch_size
        self._iter_number = 0
        self._extend_data_list = None
        self._itered_count = 0
        self.feature_list = list()
        self.extra_duplicate_number = 0
        self.mean_value = mean_value
        self.scale = scale

    def reset(self):
        ''' reset iter count to zero
        and start another data loading process'''
        self._itered_count = 0
        random.shuffle(self.feature_list)
        self._extend_data_list = self.feature_list
        for i in range(self.extra_duplicate_number):
            self._extend_data_list.append(self.feature_list[i])

    def data_precompute(self, 
                        net=None, 
                        image_size=224, 
                        epoch=1):
        '''precompute training data and save it in input path.
        Args:
            net: a dnn model to compute input features,
            image_size: input image size for training data
            epoch: training epoch
        Returns:
            None
        '''
        self._get_fixed_feature_map(image_size, 
                                    net, 
                                    epoch)
    
        self.feature_list = os.listdir(self.data_path)
        total_data_number = len(self.feature_list)
        if total_data_number % self.batch_size == 0:
            self._iter_number = int(total_data_number / self.batch_size)
        else:
            self._iter_number = int(total_data_number / self.batch_size) + 1
            self.extra_duplicate_number = self._iter_number * self.batch_size - total_data_number

    def read_next(self):
        '''read another batch of data for net training or evaluation,
        Returns:
            a list which first item is feature map and second item is label
        '''
        ret = None
        if self._itered_count < self._iter_number:
            feature_map_list = []
            label_list = []
            begin_iter = self._itered_count * self.batch_size
            end_iter = (self._itered_count + 1) * self.batch_size
            for i in range(begin_iter, end_iter):
                feature_name = os.path.join(self.data_path, self._extend_data_list[i])
                loader = F.load_as_dict(feature_name)
                feature_map = loader["F"] 
                feature_map = F.convert(feature_map, F.NCHW)
                feature_map_list.append(F.squeeze(feature_map, [0]))
                label = loader["L"] 
                #label_list.append(F.squeeze(label, [0]))
                label_list.append(label) 
            feature_map_batch = F.stack(feature_map_list, 0)
            feature_map_batch = F.convert(feature_map_batch, F.NC4HW4)
            label_batch = F.stack(label_list, 0)
            label_batch = F.reshape(label_batch, [-1])
            ret = [feature_map_batch, label_batch]
            self._itered_count += 1
        return ret

    def get_iter_number(self):
        return self._iter_number

    def _get_fixed_feature_map(self, 
                              image_size=224, 
                              fixed_net=None, 
                              epoch=1,
                              crop_ratio=0.875):
        '''save fixed feature map and its related label.
        Args:
            image_path: path to read image data
            image_size: image size for input image
            fixed_net: net for compute feature map
            is_training: whether is in trainning mode or not
            epoch: training epoch
        Returns:
            None
        '''
        self.data_path = self.train_path if self.is_training else self.val_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        else:
            pass

        dataset = ClsDataset(data_list=self.data_list,
                             input_size=image_size,
                             crop_ratio=crop_ratio,
                             is_training=self.is_training,
                             mean_value=self.mean_value,
                             scale=self.scale)
        loader = MNN_DATA.DataLoader(dataset, batch_size=1, shuffle=True)
        iter_number = loader.iter_number
        for j in range(epoch):
            epoch_name = str(j + 1) + '_epoch' 
            loader.reset()
            for i in range(iter_number):
                #example = loader.next()[0]
                example = loader.next()
                data = example[0][0]
                data = F.convert(data, F.NC4HW4)
                label = example[1][0]
                label = F.reshape(label, [-1])
                #feature_map = fixed_net.forward(data)
                feature_map = fixed_net(data)
                feature_map.fix_as_const()
                feature_map.name = 'F'
                label.name = 'L'
                data_dir = os.path.join(self.data_path , epoch_name + str(i) + '.FM')
                F.save([feature_map, label], data_dir)
