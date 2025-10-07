def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 50,
            'batch_size': 64,
            'weight_decay': 1e-3,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.975, 'im': 0.15,  'TOV_wt': 0.1},
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 50,
            'batch_size': 64,
            'weight_decay': 1e-6,
            'step_size': 50,
            'lr_decay': 0.5
        }

        self.alg_hparams = {
            'MAPU': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.375, 'im': 0.55, 'TOV_wt': 0.3},#6385
        }
class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 48,
            'weight_decay': 1e-6,
            'step_size': 50,
            'lr_decay': 0.4
        }
        self.alg_hparams = {
            'MAPU': {'pre_learning_rate': 0.002, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.1},
        }


class WISDM():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'MAPU': {'pre_learning_rate': 0.003, 'learning_rate': 0.00004, 'ent_loss_wt': 0.05897, 'im': 0.2759, 'TOV_wt': 0.9},
        }


class HHAR():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'MAPU': {'pre_learning_rate': 0.002, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759, 'TOV_wt': 0.6},
        }


