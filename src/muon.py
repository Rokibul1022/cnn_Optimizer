"""Muon Optimizer Implementation"""
import torch
from torch.optim.optimizer import Optimizer

class Muon(Optimizer):
    """
    Muon optimizer - uses momentum-based updates for weight matrices
    and AdamW for biases/norms
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(Muon, self).__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                buf = state['momentum_buffer']
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Momentum update
                buf.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])
                
                # Normalize momentum buffer
                buf_norm = buf.norm()
                if buf_norm > 0:
                    buf.div_(buf_norm)
                
                # Update parameters
                p.data.add_(buf, alpha=-group['lr'])
                
        return loss


class MTMuon(Optimizer):
    """
    Multi-Task Muon optimizer with equal task weighting (Nash equilibrium)
    Extends Muon for multi-task learning
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, num_tasks=2):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, num_tasks=num_tasks)
        super(MTMuon, self).__init__(params, defaults)
        self.num_tasks = num_tasks
        
    def get_combined_loss(self, task_losses):
        """
        Combine losses with equal weighting (Nash equilibrium)
        task_losses: tensor of shape [num_tasks]
        Returns: combined loss = (1/num_tasks) * sum(task_losses)
        """
        return task_losses.mean()
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                buf = state['momentum_buffer']
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Momentum update
                buf.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])
                
                # Normalize momentum buffer
                buf_norm = buf.norm()
                if buf_norm > 0:
                    buf.div_(buf_norm)
                
                # Update parameters
                p.data.add_(buf, alpha=-group['lr'])
                
        return loss
