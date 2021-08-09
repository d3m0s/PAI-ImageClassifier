import json

class MNNModelErrorCode(object):
    MNN_MODEL_TRAIN_FAIL = 1401
    MNN_MODEL_METRIC_FAIL = 1402
    MNN_MODEL_PREDICT_FAIL = 1403
    MNN_MODEL_DATA_CONVERT_FAIL = 1404
    MNN_MODEL_MODEL_CONVERT_FAIL = 1405
    MNN_MODEL_MODEL_INIT_FAIL = 1406

class WBReporter(object):

    @staticmethod 
    def WBPrintFunc(logName, result):
        '''
        logName: log name for related functions
        resultStr: log infos for related operation
        '''
        MNNPrefix = '[*****MNNWB*****]'
        MNNSuffix = '[-----MNNWB-----]'

        prefixFlag = MNNPrefix + '[' + logName + ']'
        suffixFlag = MNNSuffix

        resultStr = json.dumps(result)
        print('{}{}{}'.format(prefixFlag, resultStr, suffixFlag))

    @staticmethod
    def WBProcessStartReport():
        logName = 'ProcessStart'
        result = {
            "code": 0,
            "msg": "MNN process start status",
        }
        WBReporter.WBPrintFunc(logName, result)

    @staticmethod
    def WBProcessEndReport():
        logName = 'ProcessEnd'
        result = {
            "code": 0,
            "msg": "MNN process end status",
        }
        WBReporter.WBPrintFunc(logName, result)

    @staticmethod
    def reportError(code, msg):
        logName = 'ReportError' 
        result = {
            "code": code,
            "msg": msg,
        }
        WBReporter.WBPrintFunc(logName, result)
        WBReporter.WBProcessEndReport()

    @staticmethod
    def reportDataConvertStatus():
        logName = 'DataConvert'
        result = {
            "code": 0,
            "msg": "mnn model convert status",
        }
        WBReporter.WBPrintFunc(logName, result)

    @staticmethod
    def reportModelConvertStatus():
        logName = 'ModelConvert'
        result = {
            "code": 0,
            "msg": "mnn model convert status",
        }
        WBReporter.WBPrintFunc(logName, result)

    @staticmethod
    def reportPretrainStatus(class_index_dict, 
                            last_iteration,
                            max_iter=0):
        '''
        class_index_dict: a dict with key as class_name and value as index,
        laste_iteration: latest iteration for 
        '''
        logName = 'PretrainInfos'
        result = {
            "code": 0,
            "msg": "report pretrain infos",
            "data": {'class_data': str(class_index_dict),
                    'class_number': str(len(class_index_dict.keys())),
                    'last_iteration': str(last_iteration),
                    'max_iter': max_iter,
                    }
        }
        WBReporter.WBPrintFunc(logName, result)

    @staticmethod
    def reportTrainStatus(iter, 
                          maxIter, 
                          trainLoss):
        logName = 'Train'
        result = {
            "code": 0,
            "msg": "report training infos",
            "data": {
                "iter": iter,
                "maxIter": maxIter,
                "trainLoss": trainLoss,
                "percent": float(iter/maxIter) if float(iter/maxIter) <= 1 else 1,
            }
        }
        WBReporter.WBPrintFunc(logName, result)

    @staticmethod
    def reportMetricStatus(metric_value, 
                           interval):
        logName = 'Metric'
        result = {
            "code": 0,
            "msg": "report metric infos",
            "data": {
                "value": metric_value,
                "interval": interval,
            }
        }
        WBReporter.WBPrintFunc(logName, result)

    @staticmethod
    def reportPredictStatus(results, image_path, cur, max_number):
        logName = 'Predict'
        percent = (cur + 1) * 1.0 / max_number
        percent = 1 if (cur+1) == max_number else percent
        result = {
            "code": 0,
            "msg": "report predict infos",
            "data": {
                "imagePath": image_path,
                "values": results,
                "percent": percent,
            }
        }
        WBReporter.WBPrintFunc(logName, result)
