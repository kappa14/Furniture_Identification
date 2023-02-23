import torch


def save_model(epochs, model, optimizer, cost_func):

    # Save the trained model locally.
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': cost_func,
    }, 'outputs/model.pth')
