# -*- coding: UTF-8 -*-
import numpy as np
import MNN
import importlib
import cv2
import time
import os
import logging
from cores.net import Net
from cores.dataLoader import DataLoader
from cores.dataRead import DataRead
from cores.WBReporter import WBReporter
import _tools
import json

nn = MNN.nn
F = MNN.expr

class RecModel(object):
    def __init__(self, 
                pretrain_model='', 
                data_dir='',
                epoch=100, 
                last_fixed_layer_name='input', 
                input_layer_name='input', 
                image_size=224, 
                resample_epoch=5, 
                log_number = 10, 
                train_batch_size=10, 
                test_batch_size=1,
                snapshot_interval=100,
                model_type='mobilenetv3',
                save_model_path='saved_model.mnn',
                is_quantize=True):
        '''
        init process to train/evaluate a recognition model.
        Args:
            pretrain_model: path to read pretrained model
            epoch: total training epochs 
            last_fixed_layer_name: layer name for last layer to fix parameters
            input_layer_name: layer name for net input layer 
            image_size: image size for net input
            resample_epoch: resample epoch iterval
            log_number: for log interval
            train_batch_size: batch size for training process
            test_batch_size: batch size for test
            save_model_path: path to save training model
        '''
        self.save_path = os.path.join(save_model_path[:-4], 'MNNModelCache')
        self.label_txt_path = save_model_path[:-4] + '_label.txt'
        self.save_model_path = save_model_path
        self.prev_save_model = 'tmp.mnn'
        self.train_state = self._get_train_state_from_dir()
        self.mean_value = 127.5
        self.scale = 1. / 127.5
        self.data_reader = DataRead(data_dir=data_dir,
                                    cache_dir=self.save_path,
                                    batch_size=train_batch_size,
                                    mean_value=self.mean_value,
                                    scale=self.scale)
        self.class_number = len(self.data_reader.get_class_index_dict().keys())
        self.class_index_dict = self.data_reader.get_class_index_dict()
        self.net = Net(pretrain_model=pretrain_model, 
                       class_number=self.class_number, 
                       last_fixed_layer_name=last_fixed_layer_name, 
                       input_layer_name=input_layer_name,
                       model_type=model_type,
                       train_state=self.train_state)

        self.fixed_net = self.net.get_fixed_net()
        self.epoch = int(self.net.train_epoch.read()) + epoch
        self.decent_epoch_step = int(epoch / 2)
        self.log_number = log_number
        self.save_iter_number = 40
        self.resample_epoch = resample_epoch
        self.image_size = image_size
        self.train_data_path = None
        self.train_label = None
        self.is_quantize = is_quantize
        self.save_signle_train_model = True
        #for preprocess:subtrac mean and sclae to make range in (-1,1)
        #self.mean_value = [123.0, 117.0, 104.0]
        train_split_ratio = 0.8
        self.train_loader,self.test_loader = self.data_reader.get_train_val_loader(train_split_ratio) 
        self.learning_rate = 0.001
        self.fake_input = F.placeholder([1,3,self.image_size, self.image_size], F.NC4HW4)
        self.fake_input.name = 'input'
        self.opt = MNN.optim.SGD(self.net, self.learning_rate, 0.9)
        #self.opt.append(self.net.parameters)
        self._save_class_index_dict()

    def convert_dataset(self):
        '''
        convert dataset to precompute dataset
        Args:
            train_data_path: path to read training data
            train_label: path to read training label
            val_data_path: path to read validation data
            test_label: path to read test label
        Returns:
            None
        '''
        #self.train_data_path = train_data_path
        #self.train_label = train_label
        self.train_loader.data_precompute(self.fixed_net, self.image_size)
        self.test_loader.data_precompute(self.fixed_net, self.image_size)

    def predict(self, val_data_path): 
        '''
        predict process
        Args:
            val_data_path: path to read validation data
        Returns:
            None
        '''
        image_lists = os.listdir(val_data_path)
        image_len = len(image_lists)
        predict_result = dict()
        for i in range(image_len):
            if 'jpg' not in image_lists[i] and 'png' not in image_lists[i]:
                continue
            logging.info('{} image is processed'.format(i))
            image_path = os.path.join(val_data_path, image_lists[i])
            image_data = cv2.imread(image_path)
            #image_data = Image.open(image_path)
            if image_data is None:
                continue
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            image_data = cv2.resize(image_data, (self.image_size, self.image_size))
            image_data = (image_data - self.mean_value) * self.scale
            input_data = F.const(image_data.flatten().tolist(), [1, self.image_size, self.image_size, 3], F.data_format.NHWC)
            input_data = F.convert(input_data, F.NC4HW4)
            output = self.net.all_net_forward(input_data)
            output_data = output.read()
            #if use np.array, would get unexpected result
            #predict_result[image_lists[i]] = output_data
            #use list could solve this problem
            predict_result[image_lists[i]] = output_data.flatten().tolist()
        return predict_result

    def train_func(self):
        '''
        function to train a model
        '''
        self.net.train(True)
        train_epoch = 0
        #init train state with latest snapshot
        if self.train_state is not None:
            #start from next epoch num
            train_epoch = self.net.train_epoch.read() + 1
            #init learning rate with snapshot learning rate
            #self.learning_rate = self.net.learning_rate.read().tolist()
            logging.info('retrain from epoch = {}, learning rate is {}'.format(train_epoch, self.learning_rate))
        t0 = time.time()
        very_start_time = time.time()
        WBReporter.reportPretrainStatus(self.class_index_dict, train_epoch, self.epoch) 

        #start training
        for j in range(train_epoch, self.epoch):
            if j != 0 and j % self.decent_epoch_step == 0:
                self.learning_rate /= 10
                logging.info('learning_rate = {}'.format(self.learning_rate))
                self.opt = MNN.optim.SGD(self.net, self.learning_rate, 0.9)
                #self.opt.append(self.net.parameters)

            if j != 0 and (j + 1) % self.resample_epoch == 0:
                logging.info('start to resample')
                self.train_loader.data_precompute(self.fixed_net,
                                                 self.image_size)
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)

                #remove prev model to keep only latest model
                if self.save_signle_train_model:
                    if os.path.exists(self.prev_save_model):
                        os.remove(self.prev_save_model)
                tmp_save_name = os.path.join(self.save_path, str(j + 1) + '_' + 'epoch.mnn')
                self.prev_save_model = tmp_save_name
                self._save_train_state(tmp_save_name, j + 1)
                self._save_forward_net(self.save_model_path)
                accuracy = self.test_func()

            iter_number = self.train_loader.get_iter_number()
            self.train_loader.reset()
            total_loss = 0
            for i in range(0, iter_number):
                example = self.train_loader.read_next()
                data = example[0]
                label = example[1]
                #target = F.one_hot(F.cast(label, F.int), .class_number), var.float(1.0), var.float(0.0))
                target = F.one_hot(F.cast(label, F.int), self.class_number, 1, 0)
                predict = self.net.forward(data)
                loss = nn.loss.cross_entropy(predict, target)
                total_loss += loss.read()
                if i % self.log_number == 0:
                    t1 = time.time()
                    cost = t1 - t0
                    logging.info("take time cost: %.3f" % cost)
                    t0 = time.time()
                    logging.info('loss = {}'.format(loss.read()))
                self.opt.step(loss)
            #for j begin with 0, so add 1 to it as true iteration
            WBReporter.reportTrainStatus(j + 1, self.epoch, total_loss / iter_number)
        very_end_time = time.time()
        logging.info('{} epochs traning time = {}'.format(self.epoch, very_end_time - very_start_time))

    def test_func(self):
        '''
        Function to test a model
        '''
        self.net.train(False)
        self.test_loader.reset()
        iter_number = self.test_loader.get_iter_number()
        correct = 0
        total = 0
        for i in range(0, iter_number):
            example = self.test_loader.read_next()
            data = example[0]
            label = example[1]
            predict = self.net.forward(data)
            predict = F.argmax(predict, 1)
            predict = np.array(predict.read())

            label = np.array(label.read())
            correct += (np.sum(label == predict))
            total += label.size

        test_accuracy = correct / total * 100
        #test model every reample epoch
        WBReporter.reportMetricStatus(test_accuracy, self.resample_epoch)
        self.net.train(True)
        return test_accuracy

    def _get_train_state_from_dir(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        snapshots = os.listdir(self.save_path)
        train_state = None
        model_nums = len(snapshots)
        #get latest save dict:including parameters and train state
        if model_nums > 0:
            latest_index = 0
            epoch_number = 0
            for i in range(model_nums):
                if 'mnn' not in snapshots[i]:
                    continue
                index_number = int(snapshots[i].split('_')[0])
                if index_number > epoch_number:
                    epoch_number = index_number
                    latest_index = i
                file_path = os.path.join(self.save_path, snapshots[latest_index])
                self.prev_save_model = file_path
            if epoch_number != 0:
                train_state = F.load_as_list(self.prev_save_model)
                logging.info('restore train state from {}'.format(self.prev_save_model))
        return train_state

    def _save_forward_net(self, save_name):
        '''
        process to save forward net
        Args:
            save_name: name of the model to save
        Returns:
            None
        '''
        self.net.train(False)
        prob = self.net.all_net_forward(self.fake_input)
        prob.name = 'prob'
        F.save([prob], save_name)
        if self.is_quantize:
            self._mnn_weight_quantize(save_name)

    def _mnn_weight_quantize(self, model_path):
        # 4 for MNN framework
        framework_type = 3
        #last parameter: True for quantize
        _tools.mnnconvert(model_path, model_path, framework_type, False, '', True)

    def _save_train_state(self, 
                         save_name,
                         train_epoch):
        '''
        process to save training state, including model parameters and training super-params
        Args:
            save_name: name of the model to save
            train_epoch: snapshot epoch
        Returns:
            None
        '''
        train_epoch = F.const(train_epoch, [], F.NCHW, F.int)
        self.net.train_epoch = train_epoch
        #self.net.learning_rate = F.const([self.learning_rate], [])
        learning_rate = F.const([self.learning_rate], [])
        self.net.learning_rate = learning_rate
        params = self.net.parameters
        F.save(params, save_name)

    def _save_class_index_dict(self):
        json_path = os.path.join(self.save_path, 'label_map.json')
        with open(json_path, 'w') as fp:
            json.dump(self.class_index_dict, fp)
        with open(self.label_txt_path, 'w') as fp:
            for item in self.class_index_dict.keys():
                fp.write(item + '\n')
