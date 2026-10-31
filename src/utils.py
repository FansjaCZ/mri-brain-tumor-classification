import torch

def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def get_mean_std(dataset):
    channels_sum, channels_square_sum, num_batches= 0, 0, 0

    for im, _ in dataset:
        channels_sum += torch.mean(im, dim=[0,1,2])
        channels_square_sum += torch.mean(im**2, dim=[0,1,2])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_square_sum/num_batches - mean**2)

    return mean.item(), std.item()