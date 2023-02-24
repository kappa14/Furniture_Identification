import torch
import time

def save_model(epochs, model, optimizer, cost_func):

    # Save the trained model locally.
    model_name = f"outputs/model_epoch{epochs+1}.pth"
    prefix = 'model_'
    name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': cost_func,
    }, name)
