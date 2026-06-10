"""Optimizer Configurations for Multi-Task Learning"""
import torch.optim as optim
from mtadam_v2 import MTAdamV2

class OptimizerConfig:
    """Centralized optimizer configuration"""
    
    @staticmethod
    def get_config(optimizer_name, model_name, batch_size):
        """Get optimizer hyperparameters based on model, optimizer, and batch size"""
        
        # Base configurations
        configs = {
            'resnet18': {
                'sgd': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
                'sgd_nesterov': {'lr': 0.01, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 1e-4},
                'adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8},
                'adamw': {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01},
                'rmsprop': {'lr': 0.0001, 'alpha': 0.99, 'eps': 1e-8, 'momentum': 0.3, 'weight_decay': 1e-6},
                'adagrad': {'lr': 0.0001, 'lr_decay': 0, 'eps': 1e-8, 'weight_decay': 0},
                'adadelta': {'lr': 0.1, 'rho': 0.95, 'eps': 1e-6, 'weight_decay': 0},
                'mtadamv2': {'lr': 0.0003, 'num_tasks': 2, 'betas': (0.9, 0.999), 'weight_decay': 1e-6}
            },
            'mobilenetv3': {
                'sgd': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
                'sgd_nesterov': {'lr': 0.01, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 1e-4},
                'adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8},
                'adamw': {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01},
                'rmsprop': {'lr': 0.0001, 'alpha': 0.99, 'eps': 1e-8, 'momentum': 0.3, 'weight_decay': 1e-6},
                'adagrad': {'lr': 0.0001, 'lr_decay': 0, 'eps': 1e-8, 'weight_decay': 0},
                'adadelta': {'lr': 0.1, 'rho': 0.95, 'eps': 1e-6, 'weight_decay': 0},
                'mtadamv2': {'lr': 0.0003, 'num_tasks': 2, 'betas': (0.9, 0.999), 'weight_decay': 1e-6}
            }
        }
        
        # No batch size scaling - use fixed LR per optimizer
        return configs[model_name][optimizer_name].copy()
    
    @staticmethod
    def create_optimizer(optimizer_name, model_params, model_name, batch_size):
        """Create optimizer instance with proper configuration"""
        config = OptimizerConfig.get_config(optimizer_name, model_name, batch_size)
        
        if optimizer_name == 'sgd':
            return optim.SGD(model_params, **config)
        elif optimizer_name == 'sgd_nesterov':
            return optim.SGD(model_params, **config)
        elif optimizer_name == 'adam':
            return optim.Adam(model_params, **config)
        elif optimizer_name == 'adamw':
            return optim.AdamW(model_params, **config)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(model_params, **config)
        elif optimizer_name == 'adagrad':
            return optim.Adagrad(model_params, **config)
        elif optimizer_name == 'adadelta':
            return optim.Adadelta(model_params, **config)
        elif optimizer_name == 'mtadamv2':
            return MTAdamV2(model_params, **config)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    @staticmethod
    def get_loss_config(optimizer_name, model_name):
        """Get loss combination strategy"""
        # Task-balanced loss for specific optimizers
        if optimizer_name in ['rmsprop', 'adagrad', 'adadelta']:
            return {'type': 'weighted', 'cls_weight': 0.7, 'seg_weight': 0.3}
        elif optimizer_name == 'mtadamv2':
            return {'type': 'mtadam'}
        elif model_name == 'resnet18' and optimizer_name in ['adam', 'adamw']:
            return {'type': 'scaled', 'cls_scale': 0.01, 'seg_scale': 1.0}
        else:
            return {'type': 'sum'}
    
    @staticmethod
    def get_grad_clip_config(optimizer_name, model_name):
        """Get gradient clipping configuration"""
        if optimizer_name in ['rmsprop', 'adagrad', 'adadelta', 'mtadamv2']:
            return {'enabled': True, 'max_norm': 1.0}
        elif model_name == 'resnet18' and optimizer_name in ['adam', 'adamw']:
            return {'enabled': True, 'max_norm': 5.0}
        else:
            return {'enabled': False}
