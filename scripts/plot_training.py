from matplotlib import pyplot as plt
from matplotlib import ticker

from train import get_accuracy_data, get_loss_data, get_learning_rate_data

import config

from pathlib import Path
import sys

def main(save_path=str):

    if not Path(save_path).exists():
        print("Training save path does not exist")
        return

    fig, axs = plt.subplots(1,3)

    formatter = ticker.ScalarFormatter()
    formatter.set_powerlimits((-3,3))
    
    avg_loss = get_loss_data(save_path)
    axs[0].plot(range(len(avg_loss)), avg_loss, color="tab:red")
    axs[0].set_ylim(0, max(avg_loss))
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[0].yaxis.set_major_formatter(formatter)
    axs[0].grid(True)
    axs[0].set_title("Loss over training")
    axs[0].set(xlabel="Epoch", ylabel="Avarage loss")

    accuracy = get_accuracy_data(save_path)
    axs[1].plot(range(len(accuracy)), accuracy, color="tab:green")
    axs[1].set_ylim(0, 100)
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[1].yaxis.set_major_formatter(formatter)
    axs[1].grid(True)
    axs[1].set_title("Accuracy over training")
    axs[1].set(xlabel="Epoch", ylabel="Avarage accuracy")

    learning_rate = get_learning_rate_data(save_path)
    axs[2].plot(range(len(learning_rate)), learning_rate, color="tab:blue")
    axs[2].set_ylim(0, max(learning_rate))
    axs[2].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[2].yaxis.set_major_formatter(formatter)
    axs[2].grid(True)
    axs[2].set_title("Learning rate over training")
    axs[2].set(xlabel="Epoch", ylabel="Learning rate")

    plt.tight_layout()
    plt.show()

    print(f"Close the opened matplot lib window to stop the script.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Not enough arguments")