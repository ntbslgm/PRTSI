import os
import torch


def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.dropout = 0.2
        self.kernel_size = 30
        self.stride = 1
        # added
        self.input_channels_add = 31
        # features
        self.mid_channels = 16
        self.final_out_channels = 8
        self.features_len = 65 # for my model

        # ICB ASB
        self.AR_hid_dim = 8
        self.dim = 65 # change 65
        self.in_features = 377 # change 8 -> 749

        # tAPE
        self.fea_len = 3000
        self.d_model = 128

class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 5120
        self.scenarios = [("0", "1"), ("1", "2"), ("3", "1"), ("1", "0"), ("2", "3")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # ICB ASB
        self.AR_hid_dim = 128
        self.dim = 109
        self.in_features = 109   # change 128 -> 1279

        # tAPE
        self.fea_len = 5120
        self.d_model = 128

class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16"),]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 12
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6
        # added
        self.input_channels_add = 31

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # ICB ASB
        self.AR_hid_dim = 128
        self.dim = 31
        self.in_features = 18

        # tAPE
        self.fea_len = 128
        self.d_model = 128

class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.scenarios = [("7", "18"), ("20", "30"), ("35", "31"), ("17", "23"), ("6", "19"),
                          ("2", "11"), ("33", "12"), ("5", "26"), ("28", "4"), ("23", "32")]
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # ICB ASB
        self.AR_hid_dim = 128
        self.dim = 31
        self.in_features = 18

        # tAPE
        self.fea_len = 128
        self.d_model = 128

class HHAR(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR, self).__init__()
        self.sequence_len = 128
        self.scenarios = [("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5"),
                          ]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # ICB ASB
        self.AR_hid_dim = 128
        self.dim = 31
        self.in_features = 18

        # tAPE
        self.fea_len = 128
        self.d_model = 128