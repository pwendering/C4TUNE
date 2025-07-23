# -*- coding: utf-8 -*-
"""
Functionalities for saving and loading checkpoints
"""

import torch


def checkpoint(epoch, model, optimizer, scheduler, train_loss, test_losses,
               loss_fn, path):
    '''

    Parameters
    ----------
    epoch : int
        current epoch
    model : model.NeuralNetwork < nn.Module
        neural network object
    optimizer : torch.optim.Optimizer
        optimizer for model training
    scheduler : torch.optim.lr_scheduler.LRScheduler
        learning rate scheduler
    train_loss : float
        training loss
    test_losses : float
        test losses
    loss_fn: str
        loss function
    path : str
        path where checkpoint should be stored

    Returns
    -------
    None.

    '''
    
    rng_state = torch.random.get_rng_state()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'test_losses': test_losses,
        'loss_fn': loss_fn,
        'rng_state': rng_state
        }, path)


def resume(model, optimizer, scheduler, path):
    '''
    
    Parameters
    ----------
    model : model.NeuralNetwork < nn.Module
        neural network object
    optimizer : torch.optim.Optimizer
        optimizer for model training
    scheduler : torch.optim.lr_scheduler.LRScheduler
        learning rate scheduler
    path : str
        path where checkpoint should be stored

    Returns
    -------
    epoch : int
        resume epoch
    train_loss : float
        training loss
    test_losses : float
        test losses
    loss_fn : str
        loss function

    '''
    
    cpoint = torch.load(path, weights_only=True)
    
    model.load_state_dict(cpoint['model_state_dict'])
    optimizer.load_state_dict(cpoint['optimizer_state_dict'])
    scheduler.load_state_dict(cpoint['scheduler_state_dict'])
    torch.set_rng_state(cpoint['rng_state'])
    
    epoch = cpoint['epoch']
    train_loss = cpoint['train_loss']
    test_losses = cpoint['test_losses']
    loss_fn = cpoint['loss_fn']
    
    return epoch, train_loss, test_losses, loss_fn