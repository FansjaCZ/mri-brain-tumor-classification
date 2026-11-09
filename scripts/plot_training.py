from matplotlib import pyplot as plt
from matplotlib import ticker
from sklearn import metrics

from train import get_training_data

from pathlib import Path
import math
import sys

def main(save_path=str):

    if not Path(save_path).exists():
        print("Training save path does not exist")
        return

    training_data = get_training_data(save_path)

    formatter = ticker.ScalarFormatter()
    formatter.set_powerlimits((-3,3))

    fig, axs = plt.subplots(2,2)

    axs[0, 0].plot(range(len(training_data["train_loss"])), training_data["train_loss"], label="Training")
    axs[0, 0].plot(range(len(training_data["test_loss"])), training_data["test_loss"], label="Evaluation")
    axs[0, 0].legend()
    axs[0, 0].set_ylim(0, max(training_data["train_loss"]+training_data["test_loss"]))
    axs[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(math.ceil(len(training_data["test_loss"])/10)))
    axs[0, 0].yaxis.set_major_formatter(formatter)
    axs[0, 0].grid(True)
    axs[0, 0].set_title("Loss over training")
    axs[0, 0].set(xlabel="Epoch", ylabel="Loss")

    axs[0, 1].plot(range(len(training_data["train_accuracy"])), training_data["train_accuracy"], label="Training")
    axs[0, 1].plot(range(len(training_data["test_accuracy"])), training_data["test_accuracy"], label="Evaluation")
    axs[0, 1].legend()
    axs[0, 1].set_ylim(0, 100)
    axs[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(math.ceil(len(training_data["test_accuracy"])/10)))
    axs[0, 1].yaxis.set_major_formatter(formatter)
    axs[0, 1].grid(True)
    axs[0, 1].set_title("Accuracy over training")
    axs[0, 1].set(xlabel="Epoch", ylabel="Accuracy")

    axs[1, 0].plot(range(len(training_data["learning_rate"])), training_data["learning_rate"])
    axs[1, 0].set_ylim(0, max(training_data["learning_rate"]))
    axs[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(math.ceil(len(training_data["learning_rate"])/10)))
    axs[1, 0].yaxis.set_major_formatter(formatter)
    axs[1, 0].grid(True)
    axs[1, 0].set_title("Learning rate over training")
    axs[1, 0].set(xlabel="Epoch", ylabel="Learning rate")

    axs[1, 1].set_title(f"Confusion matrix ")
    ds = metrics.ConfusionMatrixDisplay(training_data["confusion_matrix"].detach().contiguous().numpy(), display_labels=["glioma", "meningioma", "notumor", "pituitary"])
    ds.plot(ax=axs[1, 1], cmap="Blues", colorbar=False)

    plt.tight_layout()
    plt.show()

    print(f"Close the opened matplot lib windows or stop the script to exit.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Not enough arguments")