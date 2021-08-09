# -*- coding: UTF-8 -*-
import MNN
import logging

nn = MNN.nn
F = MNN.expr

class Net(nn.Module):
    def __init__(self, 
                 pretrain_model=None, 
                 class_number=1000, 
                 last_fixed_layer_name='input', 
                 input_layer_name='input',
                 model_type='mobilenetv3_small',
                 train_state=None):
        '''init process for fast finetuning a model.
        Args:
            pretrain_model: path to get pretrain_model
            class_number: number of classes for training task
            last_fixed_layer_name: layer name for last layer to fix parameters
            input_layer_name: layer name for input layer
            output_layer_name: layer name for output layer
        '''
        super(Net, self).__init__()
        self.learning_rate = F.const([0.], [])
        self.train_epoch = F.const([0], [], F.NCHW, F.int)
        self.train_state = train_state
        #if have snapshot, restore from latest ckpt
        var_map = F.load_as_dict(pretrain_model)
        logging.info('restore model from {}'.format(pretrain_model))
        input_var = var_map[input_layer_name]
        last_fixed_layer_var = var_map[last_fixed_layer_name]
        first_train_var = var_map[last_fixed_layer_name]
        self.fixed_net = nn.load_module([input_var], [last_fixed_layer_var], False)
        if model_type == 'mobilenetv2':
            output_layer_name='MobilenetV2/Logits/AvgPool'
        else:
            output_layer_name='MobilenetV3/Logits/AvgPool'
        output_var = var_map[output_layer_name]
        self.net = nn.load_module([first_train_var], [output_var], True)
        #for init train only
        if model_type == 'mobilenetv3_small':
            expand_number = 1024
        else:
            expand_number = 1280

        self.fc = nn.conv(expand_number, class_number, [1, 1])
        #define the name of fc layer as logit
        #self.fc.name = 'logit'
        self.fc.set_name('logit')
        
        #if have snapshot, restore from latest ckpt
        if self.train_state is not None:
            logging.info('init model with snapshot params')
            self.load_parameters(self.train_state)

    def forward(self, x):
        '''
        net forward process
        Args:
            x: input data for net
        Returns:
            a tensor: net output
        '''
        x = self.net(x)
        #init from imageNet pretrain model, fc layers is random initialized
        x = self.fc(x)
        x = F.softmax(F.reshape(F.convert(x, F.NCHW), [0, -1]))
        return x

    def all_net_forward(self, x):
        '''compute entire dnn net, including fixed net.
        Args:
            x: input data for net
        Returns:
            a tensor: net output
        '''
        x = self.fixed_net(x)
        x = self.net(x)
        x = self.fc(x)
        x = F.softmax(F.reshape(F.convert(x, F.NCHW), [0, -1]))
        return x

    def get_fixed_net(self):
        return self.fixed_net

    def fixed_net_forward(self, x):
        return self.fixed_net(x)
