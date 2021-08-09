# -*- coding: UTF-8 -*-
import MNN
F = MNN.expr
nn = MNN.nn
class Net(nn.Module):
    def __init__(self, 
                 pretrain_model, 
                 output_layer_names=None,
                 input_layer_name='input'):
        super(Net, self).__init__()
        var_map = F.load_as_dict(pretrain_model)
        inputs_outputs = F.get_inputs_and_outputs(var_map)
        input_vars = []
        output_vars = []
        if isinstance(output_layer_names, list):
            for name in output_layer_names:
                output_vars.append(var_map[name])
        else:
            output_vars.append(var_map[output_layer_names])
        input_vars.append(var_map[input_layer_name])
        self.net = nn.load_module(input_vars, output_vars, False)

    def forward(self, x):
        x = self.net.forward(x)
        return x
