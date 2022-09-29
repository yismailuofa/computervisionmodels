import torch
import numpy as np


def knn(x_train, y_train, x_test, n_classes, device):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 1000 x 784 testing images
    n_classes: no. of classes in the classification task
    device: pytorch device on which to run the code
    return: predicted y_test which is a 1000-sized vector
    """
    k = 7
    NUM_TESTS = x_test.shape[0]

    # convert numpy arrays to pytorch tensors, scaled by 255 color range
    x_train = torch.tensor(x_train / 255, dtype=torch.float, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    x_test = torch.tensor(x_test / 255, dtype=torch.float, device=device)
    y_test = np.zeros(NUM_TESTS)

    for i, row in enumerate(x_test):
      # find the distances with each row in the test data
      distances = torch.linalg.norm(x_train - row, dim=1)
      
      # find the indices of the closest samples
      minDistanceIndices = torch.topk(distances,k,largest=False).indices

      # get the prediction with the highest frequency
      oneHotLabels = torch.nn.functional.one_hot(y_train[minDistanceIndices],n_classes)
      prediction = torch.argmax(torch.sum(oneHotLabels, 0)).item()

      y_test[i] = prediction
    
    return y_test
