import numpy as np
import random
import MNN
import cv2
F = MNN.expr

class ClsDataset(MNN.data.Dataset):
    def __init__(self, 
                 data_list=list(),
                 input_size=224,
                 crop_ratio=0.875,
                 is_training=False,
                 mean_value=127.5,
                 scale=1./127.5):
        super(ClsDataset, self).__init__()
        self.data_list=data_list
        self.input_size = input_size
        self.is_training = is_training
        self.mean_value = mean_value
        self.crop_ratio = crop_ratio
        self.scale = scale

    def __getitem__(self, index):
        total_image_number = len(self.data_list)
        if index < total_image_number:
            data_dict = self.data_list[index]
            image_name = [key for key in data_dict.keys()]
            image_name = image_name[0]
            image_label = int(data_dict[image_name])
            image_data = cv2.imread(image_name)
            # print('[~~~] read image at {}, {}'.format(image_name, image_data))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            image_data = self._cls_preprocess(image_data)

            dv = F.const(image_data.flatten().tolist(), [self.input_size, self.input_size, 3], F.data_format.NHWC)
            dl = F.const([image_label], [], F.data_format.NHWC, F.dtype.int)
            # first for inputs, and may have many inputs, so it's a list
            # second for targets, also, there may be more than one targets
            return [dv], [dl]
        else:
            return None,None

    def __len__(self):
        # size of the dataset
        return len(self.data_list)

    def _cls_preprocess(self, image):
        '''
        training:including random center crop, image resize and random flip left, right
        test: only resize involved
        '''

        height,width,_ = image.shape
        crop_height = height
        crop_width = width
        ret_image = None

        if self.is_training:
            crop_height = int(height * self.crop_ratio)
            crop_width = int(width * self.crop_ratio)
            x = random.randint(0, width - crop_width)
            y = random.randint(0, height - crop_height)
            crop_image = image[y:y+crop_height, x:x+crop_width]
            ret_image = cv2.resize(crop_image, (self.input_size, self.input_size))

            #random flip image left to right
            if random.randint(0, 1):
            #    #ret_image = np.fliplr(ret_image)
            #    #ret_image = np.fliplr(ret_image).astype('float64')
                ret_image = cv2.flip(ret_image, 1)
        else:
            ret_image = cv2.resize(image, (self.input_size, self.input_size))
        ret_image = (ret_image - self.mean_value) * self.scale

        return ret_image
