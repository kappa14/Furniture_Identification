import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')


def save_plots(train_acc, valid_acc, train_loss, valid_loss):

    # Save the loss and accuracy plots locally.

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='Train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='Validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Accuracy.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='Train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='Validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Loss.png')
