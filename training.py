import torch
from tqdm import tqdm
from FISHClass.utils.device import best_gpu
from FISHClass.utils.training import TrainingScheduler

def train(epochs, model, training_loader, validation_loader, loss_fn, optimizer, save_dir, scheduler=None, patients=50, use_gpu=True, device=None):
    
    model_saver = TrainingScheduler(model, optimizer, save_dir, patients)
    
    if isinstance(device, type(None)):
        device = torch.device(best_gpu() if torch.cuda.is_available() and use_gpu else "cpu")
    
    model.to(device)
    
    ret_values = []
    
    with tqdm(range(epochs)) as epoch_loop:
    
        for epoch in epoch_loop:
            
            if model_saver.stop_early:
                break
            
            train_ret = model.train_fn(training_loader, loss_fn, optimizer, scheduler, device)

            val_ret = model.validation_fn(validation_loader, loss_fn, device)
                        
            model_saver.add_loss(epoch, train_ret["train_loss"], val_ret["val_loss"], val_ret["accuracy"] if "accuracy" in val_ret.keys() else None)

            epoch_loop.set_postfix({**val_ret, **train_ret})
            ret_values.append({**val_ret, **train_ret})
        
    return ret_values