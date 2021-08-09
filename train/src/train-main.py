import numpy as np
from cores.model import RecModel
from cores.WBReporter import WBReporter
import logging
import os
import sys
import time
from threading import Thread

MNNGlobalVars = {}
MNNGlobalVars["epoch"] = 5
MNNGlobalVars["quant"] = True
MNNGlobalVars["sessionName"] = "476a4f577ea84fa1a77fbf601d815bcc"

is_remote = True
if os.uname().sysname == 'Darwin':
    print('[PAI] using local train for uname', os.uname())
    is_remote = False
else:
    print('[PAI] using remote train for uname', os.uname())

dir_path = os.path.dirname(os.path.realpath(__file__))
print("[PAI] dir path", dir_path)
resource_path = os.path.join(dir_path, '..', 'resource')

MNNGlobalVars["savePath"] = os.path.join(dir_path, '..', 'result', 'model.mnn')
MNNGlobalVars["trainDatasetPath"] = os.path.join(resource_path, 'dataset/ImageClassifier')
    

pretrain_model_path = os.path.join(resource_path, 'pretrain_model/mb3_small.mnn')
print('checkpath resource_path={}, model_path={}, dataset_path={}, save_path={}'.format(resource_path, pretrain_model_path, MNNGlobalVars["trainDatasetPath"], MNNGlobalVars["savePath"]))

def train():
    global pretrain_model_path
    '''
    training and validation process for finetune a model
    '''
    #LocalGlobalVars = {
    #        'pretrain_model': "./pretrain_model/mb3.mnn",
    #        'last_fixed_layer_name': "MobilenetV3/expanded_conv_14/add",
    #        'input_layer_name': "input",
    #         'snapshot_every_n_epochs': 5,
    #         'model_type': 'mobilenetv3'
    #        }
    LocalGlobalVars = {
            'pretrain_model': pretrain_model_path,
            'last_fixed_layer_name': "MobilenetV3/expanded_conv_10/add",
            'input_layer_name': "input",
             'snapshot_every_n_epochs': 5,
             'model_type': 'mobilenetv3_small'
            }
    logging.info('start init model process:')

    if 'trainDatasetPath' in MNNGlobalVars and 'savePath' in MNNGlobalVars:
        epoch = 0
        if 'epoch' in MNNGlobalVars:
            epoch = int(MNNGlobalVars['epoch'])
        is_quantize = True
        if 'quant' in MNNGlobalVars:
            is_quantize = MNNGlobalVars['quant']
        #report train process start
        WBReporter.WBProcessStartReport()
        model = RecModel(pretrain_model=LocalGlobalVars['pretrain_model'], 
                         data_dir=MNNGlobalVars['trainDatasetPath'],
                         epoch=epoch,
                         last_fixed_layer_name=LocalGlobalVars['last_fixed_layer_name'],
                         input_layer_name=LocalGlobalVars['input_layer_name'],
                         model_type=LocalGlobalVars['model_type'],
                         save_model_path=MNNGlobalVars['savePath'],
                         snapshot_interval=LocalGlobalVars['snapshot_every_n_epochs'],
                         is_quantize=is_quantize)

        logging.info('start to convert dataset:')
        model.convert_dataset()
        WBReporter.reportDataConvertStatus()

        logging.basicConfig(level = logging.INFO)

        logging.info('start train process:')
        model.train_func()

    #report end of train process
    WBReporter.WBProcessEndReport()

class AutoFlusing:
    @staticmethod
    def autoflush(delta=0.5):
        return AutoFlusing(delta)

    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.running = False

    def __enter__(self):
        self.running = True
        # print('===> Enter')

        def flushing():
            while self.running:
                time.sleep(self.delta)
                sys.stdout.flush()
                sys.stderr.flush()

        Thread(target=flushing).start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print('<=== Leave')
        self.running = False

if __name__ == '__main__':
    if is_remote:
        train()
    else:
        with AutoFlusing.autoflush():
            train()

