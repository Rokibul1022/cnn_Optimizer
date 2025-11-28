"""MTAdamV2: Equal Weighting Multi-Task Optimizer"""
import torch
from torch.optim import Adam

class MTAdamV2(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, num_tasks=2, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.num_tasks = num_tasks
        self.task_weights = torch.ones(num_tasks) / num_tasks
    
    def get_combined_loss(self, task_losses):
        """Combine losses with equal weighting (0.5/0.5)"""
        if isinstance(task_losses, list):
            task_losses = torch.stack(task_losses)
        w = self.task_weights.to(task_losses.device, dtype=task_losses.dtype)
        return (w * task_losses).sum()
    
    def get_task_weights(self):
        return self.task_weights.cpu().numpy()
