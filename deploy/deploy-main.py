#!/usr/bin/python
# -*- coding: UTF-8 -*-
import MNN
import MNNCV
import cv2
import os
import numpy as np
from net import Net

import time
F = MNN.expr
kit = MNNCV.Kit()
vars = kit.getEnvVars()
work_path = vars['workPath']
print('[+] deploy workpath', work_path)

#To replace MNN_PROJECT_NAME with related project name
model_path = os.path.join(work_path, 'resource', 'result' ,'model.mnn')
global_label_path = os.path.join(work_path, 'resource', 'result', 'model_label.txt')
output_layer_names = 'prob'
input_layer_name = 'input'
global_net = Net(model_path,
                output_layer_names,
                input_layer_name)

class ClassifierPlayground(object): 
    @staticmethod
    def get_label_map_dict(label_path):
        label_map_dict = {}
        with open(label_path, 'r') as fi:
            cont = fi.readlines()
            for i,class_name in enumerate(cont):
                label_map_dict[i] = class_name
        return label_map_dict

    @staticmethod
    def run():
        kit.open("image")
        kit.selectPhoto(ClassifierPlayground.selectPhotoCallback)
        kit.setCloseCallback(ClassifierPlayground.closeCallback)

    @staticmethod
    def selectPhotoCallback(data, format, imgw, imgh):
        class_name,max_score = ClassifierPlayground.mnn_image_predict(data)
        label_map_dict = ClassifierPlayground.get_label_map_dict(global_label_path)
        kit.setLabel(0, 'name:{},score:{}'.format(label_map_dict[class_name], max_score), {"color":"#FF0000"})

    @staticmethod 
    def mnn_image_predict(orin_image_data, image_format=0):
        image_size = 224
        if orin_image_data is not None:
            if image_format == 0:
                image_data = cv2.cvtColor(orin_image_data, cv2.COLOR_RGBA2RGB)                
            elif image_format == 2:
                image_data = cv2.cvtColor(orin_image_data, cv2.COLOR_BGR2RGB)
            elif image_format == 3:
                image_data = cv2.cvtColor(orin_image_data, cv2.COLOR_GRAY2RGB)
            elif image_format == 4:
                image_data = cv2.cvtColor(orin_image_data, cv2.COLOR_BGRA2RGB)
            elif image_format == 11:
                image_data = cv2.cvtColor(orin_image_data, cv2.COLOR_YUV2RGB_NV21)
            elif image_format == 12:
                image_data = cv2.cvtColor(orin_image_data, cv2.COLOR_YUV2RGB_NV12)
            else:
                image_data = orin_image_data
            image_data = cv2.resize(image_data, (image_size, image_size))
            image_height, image_width, _ = orin_image_data.shape
            normalized_image_data = (2.0 / 255.0) * image_data - 1.0
            input_data = F.placeholder([1, image_size, image_size, 3], F.NHWC, F.float)
            input_data.write(normalized_image_data.tolist())
            input_data = F.convert(input_data, F.NC4HW4)
            outputs = global_net.forward([input_data])[0]
            output_data = outputs.read()
            #if use np.array, would get unexpected result
            #predict_result[image_lists[i]] = output_data
            output_data = np.array(output_data.flatten().tolist())
            scores = np.max(output_data)
            #class_name = self.label_map_dict[np.argmax(output_data)]
            class_name = np.argmax(output_data)
            return [class_name, scores]
    
    def closeCallback(self, key):
        print "End of Callback"

if __name__ == '__main__':
   ClassifierPlayground.run()
