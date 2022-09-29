# -*- coding: utf-8 -*-
"""Assignment-3 Logistic Regression
CCID: yismail
Name: Youssef Ismail
Student ID: 1616494
"""

import torch
from torchvision import transforms, datasets
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
from torch.utils.data.sampler import *
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
            * 100
    return base_score


def run(algorithm, dataset_name, filename):
    start = timeit.default_timer()
    predicted_test_labels, gt_labels = algorithm(dataset_name)
    if predicted_test_labels is None or gt_labels is None:
        return (0, 0, 0)
    stop = timeit.default_timer()
    run_time = stop - start

    np.savetxt(filename, np.asarray(predicted_test_labels))

    correct = 0
    total = 0
    for label, prediction in zip(gt_labels, predicted_test_labels):
        total += label.size(0)
        correct += (prediction.cpu().numpy() == label.cpu().numpy()
                    ).sum().item()   # assuming your model runs on GPU

    accuracy = float(correct) / total

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
    return (correct, accuracy, run_time)


class MultipleLinearRegression(torch.nn.Module):
    def __init__(self, inFeatures):
        super(MultipleLinearRegression, self).__init__()
        # 10 output features for both datasets
        self.f = torch.nn.Linear(inFeatures, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.f(x)


nEpochs = 4
trainBatchSize = 120
testBatchSize = 1000
validateOnly = False

lr = 1e-3  # learning rate
lam = 1e-3  # lambda
opt = "ADAM"  # optimizer


def logistic_regression(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model & optimizer with L2 regularization in weight_decay
    model = MultipleLinearRegression(
        28*28 if dataset_name == "MNIST" else 3*32*32).to(device)
    if opt == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=lam)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=lam)

    print(f"Initialization with lr:{lr} lam:{lam} opt:{opt}")

    # Get the datasets
    if (dataset_name == "MNIST"):
        training = datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))]))
        test = datasets.MNIST('/MNIST_dataset/', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]))

        training, validation = torch.utils.data.random_split(training, [
                                                             48000, 12000])
    else:
        training = datasets.CIFAR10(root='/CIFAR10_dataset', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test = datasets.CIFAR10(root='/CIFAR10_dataset', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        training, validation = torch.utils.data.random_split(training, [
                                                             38000, 12000])

    # Initialize the loaders
    trainingLoader = torch.utils.data.DataLoader(training,
                                                 batch_size=trainBatchSize,
                                                 shuffle=True, num_workers=2)
    validationLoader = torch.utils.data.DataLoader(validation,
                                                   batch_size=trainBatchSize,
                                                   shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(test,
                                             batch_size=testBatchSize,
                                             shuffle=False, num_workers=2)

    def train(epoch):
        model.train()

        for batch, (data, target) in enumerate(trainingLoader):
            # load data on device
            data = data.to(device)
            target = target.to(device)

            # zero out and predict with the data
            optimizer.zero_grad()
            output = model(data)

            # calculate the loss with L2 term
            crossLoss = torch.nn.CrossEntropyLoss()
            loss = crossLoss(output, target)

            # adjust based on loss
            loss.backward()
            optimizer.step()
            if (not batch % 100):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch * len(data), len(trainingLoader.dataset),
                    100. * batch / len(trainingLoader), loss.item()))

    def validate():
        model.eval()  # put in eval mode
        loss = 0
        correct = 0

        # Used for hyperparameter tuning
        predictions = torch.empty(size=(len(validationLoader), trainBatchSize))
        labels = torch.empty(size=(len(validationLoader), trainBatchSize))

        with torch.no_grad():
            for i, (data, target) in enumerate(validationLoader):
                # load data on device
                data = data.to(device)
                target = target.to(device)

                # predict with the data, choose best of 10 outputs
                output = torch.nn.functional.softmax(model(data), dim=1)
                pred = output.data.max(1, keepdim=True)[1]

                # calculate correct and loss
                correct += pred.eq(target.data.view_as(pred)).sum()
                crossLoss = torch.nn.CrossEntropyLoss(reduction='sum')
                loss += crossLoss(output, target).item()

                if (validateOnly):
                    predictions[i] = pred.reshape(-1)
                    labels[i] = target.data

        loss /= len(validationLoader.dataset)
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(validationLoader.dataset),
            100. * correct / len(validationLoader.dataset)))

        if validateOnly:
            return predictions, labels

    def test():
        model.eval()  # put in eval mode
        loss = 0
        correct = 0
        predictions = torch.empty(size=(len(testLoader), testBatchSize))
        labels = torch.empty(size=(len(testLoader), testBatchSize))

        with torch.no_grad():
            for i, (data, target) in enumerate(testLoader):
                # load data on device
                data = data.to(device)
                target = target.to(device)

                # predict with the data, choose best of 10 outputs
                output = torch.nn.functional.softmax(model(data), dim=1)
                pred = output.data.max(1, keepdim=True)[1]

                # calculate correct and loss
                correct += pred.eq(target.data.view_as(pred)).sum()
                crossLoss = torch.nn.CrossEntropyLoss(reduction='sum')
                loss += crossLoss(output, target).item()

                # Store the batch in the output
                predictions[i] = pred.reshape(-1)
                labels[i] = target.data

        loss /= len(testLoader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(testLoader.dataset), 100. * correct / len(testLoader.dataset)))

        return predictions, labels

    for epoch in range(1, nEpochs + 1):
        train(epoch)
        if not validateOnly:
            validate()
    if validateOnly:
        return validate()
    return test()


def tune_hyper_parameter():
    start = timeit.default_timer()

    lrVals = [1e-3, 1e-2]
    lambdaVals = [1e-3, 1e-2]
    optimizerVals = ["SGD", "ADAM"]
    params = ((lr, lam, opt)
              for lr in lrVals for lam in lambdaVals for opt in optimizerVals)

    bestParams = None
    bestAcc = 0.0
    global validateOnly
    validateOnly = True

    for lr_, lam_, opt_ in params:
        global lr, lam, opt
        lr, lam, opt = lr_, lam_, opt_

        res, _ = run_on_dataset(
            "CIFAR10", "predictions_cifar10_YoussefIsmail_1616494.txt")
        currAcc = res["accuracy"]
        if currAcc > bestAcc:
            bestAcc = currAcc
            bestParams = (lr_, lam_, opt_)

    validateOnly = False

    return bestParams, bestAcc, timeit.default_timer() - start


"""Main loop. Run time and total score will be shown below."""


def run_on_dataset(dataset_name, filename):
    if dataset_name == "MNIST":
        min_thres = 0.82
        max_thres = 0.92

    elif dataset_name == "CIFAR10":
        min_thres = 0.28
        max_thres = 0.38

    correct_predict, accuracy, run_time = run(
        logistic_regression, dataset_name, filename)

    score = compute_score(accuracy, min_thres, max_thres)
    result = OrderedDict(correct_predict=correct_predict,
                         accuracy=accuracy, score=score,
                         run_time=run_time)
    return result, score


def main():
    filenames = {"MNIST": "predictions_mnist_YoussefIsmail_1616494.txt",
                 "CIFAR10": "predictions_cifar10_YoussefIsmail_1616494.txt"}
    result_all = OrderedDict()
    score_weights = [0.5, 0.5]
    scores = []
    for dataset_name in ["MNIST", "CIFAR10"]:
        result_all[dataset_name], this_score = run_on_dataset(
            dataset_name, filenames[dataset_name])
        scores.append(this_score)
    total_score = [score * weight for score,
                   weight in zip(scores, score_weights)]
    total_score = np.asarray(total_score).sum().item()
    result_all['total_score'] = total_score
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result_all, indent=4))
    print("\nResult:\n", pformat(result_all, indent=4))


main()
