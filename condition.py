import torch.nn.functional as F
import torch.nn

def get_one_hot_labels(labels, num_classes):
    return F.one_hot(labels, num_classes)

def combine_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), 1)
    return combined