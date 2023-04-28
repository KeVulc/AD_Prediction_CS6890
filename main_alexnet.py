import random
import torch
import numpy as np

import argparse
import logging

import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image

import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter

from custom_transform2D import CustomResize
from custom_transform2D import CustomToTensor

from AD_Standard_2DSlicesData import AD_Standard_2DSlicesData
from AD_Standard_2DRandomSlicesData import AD_Standard_2DRandomSlicesData

from AlexNet2D import alexnet
from AlexNet2D_SE import alexnet_se
from AlexNet2D_SE_topology import alexnet_se_topology
from metrics import getData, reportMetrics
import metrics
from random_splits import random_split


import numpy as np
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for JHU CS661 Computer Vision HW3.")

parser.add_argument("--load",
                    help="Load saved network weights.")
parser.add_argument("--save", default="AlexNet",
                    help="Save network weights.")
parser.add_argument("--augmentation", default=True, type=bool,
                    help="Save network weights.")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[0], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")


# feel free to add more arguments as you need


def main(options):
    # Path configuration
    TRAINING_PATH = 'split_train.txt'
    VALIDATION_PATH = 'split_validation.txt'
    TESTING_PATH = 'split_test.txt'
    IMG_PATH = './Image'

    SEED = 11
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    
    # model_name = 'AlexNet2D_Pretrained'
    # model_name = 'AlexNet2D_SE_Pretrained'
    # model_name = 'AlexNet2D_SE_Topology'
    model_name = 'AlexNet2D_Topology'

    random_split(SEED, 336, 84, 83, 'subjects.txt', TRAINING_PATH, VALIDATION_PATH, TESTING_PATH)

    trg_size = (224, 224)

    transformations = transforms.Compose([CustomResize(trg_size),
                                          CustomToTensor()
                                        ])
    
    dset_train = AD_Standard_2DRandomSlicesData(IMG_PATH, TRAINING_PATH, transformations)
    dset_valid = AD_Standard_2DSlicesData(IMG_PATH, VALIDATION_PATH, transformations)
    dset_test = AD_Standard_2DSlicesData(IMG_PATH, TESTING_PATH, transformations)

    train_loader = DataLoader(dset_train,
                                batch_size=options.batch_size,
                                shuffle=False,
                                num_workers=4,
                                drop_last=True
                            )

    valid_loader = DataLoader(dset_valid,
                             batch_size=options.batch_size,
                             shuffle=False,
                             num_workers=4,
                             drop_last=False
                            )

    test_loader = DataLoader(dset_test,
                             batch_size=options.batch_size,
                             shuffle=True,
                             num_workers=4,
                             drop_last=False
                            )

    use_cuda = (len(options.gpuid) >= 1)
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    # Initial the model
    if model_name == 'AlexNet2D_Pretrained':
        model = alexnet(pretrained=True)
    elif model_name == 'AlexNet2D_SE_Pretrained':
        model = alexnet_se(pretrained=True)
    elif model_name == 'AlexNet2D_SE_Topology':
        model = alexnet_se_topology()
    elif model_name == 'AlexNet2D_Topology':
        model = alexnet(pretrained=False)


    # model.load_state_dict(torch.load(options.load))

    if use_cuda > 0:
        model.cuda()
    else:
        model.cpu()

    # Binary cross-entropy loss
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()

    lr = options.learning_rate
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)
    best_f1 = float("-inf")

    train_loss_f = open("train_loss.txt", "w")
    train_acc_f = open("train_accuracy.txt", "w")
    train_pre_f = open("train_precision.txt", "w")
    train_rec_f = open("train_recall.txt", "w")
    train_f1_f = open("train_f1.txt", "w")
    valid_acu_f = open("valid_accuracy.txt", "w")
    valid_pre_f = open("valid_precision.txt", "w")
    valid_rec_f = open("valid_recall.txt", "w")
    valid_f1_f = open("valid_f1.txt", "w")

    for epoch_i in range(options.epochs):

        logging.info("At {0}-th epoch.".format(epoch_i))
        train_loss, data_train = train(model, train_loader, use_cuda, criterion, optimizer)
        # each instance in one batch has 3 views
        train_avg_loss = train_loss / (len(dset_train) * 3 / options.batch_size)
        logging.info("Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss.data.item(), epoch_i))
        logging.info("Average training accuracy is {0:.5f} at the end of epoch {1}".format(metrics.accuracy(data_train), epoch_i))
        train_loss_f.write("{0:.5f}\n".format(train_avg_loss))
        train_acc_f.write("{0:.5f}\n".format(metrics.accuracy(data_train)))
        train_rec_f.write("{0:.5f}\n".format(metrics.recall(data_train)))
        train_pre_f.write("{0:.5f}\n".format(metrics.precision(data_train)))
        train_f1_f.write("{0:.5f}\n".format(metrics.F1(data_train)))

        data = validate(model, valid_loader, use_cuda)

        logging.info("Average validation accuracy is {0:.5f} at the end of epoch {1}".format(metrics.accuracy(data), epoch_i))

        # write validation accuracy to file
        valid_acu_f.write("{0:.5f}\n".format(metrics.accuracy(data)))
        valid_rec_f.write("{0:.5f}\n".format(metrics.recall(data)))
        valid_pre_f.write("{0:.5f}\n".format(metrics.precision(data)))
        dev_f1 = metrics.F1(data)
        valid_f1_f.write("{0:.5f}\n".format(dev_f1))

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), open(options.save, 'wb'))
            best_data_valid = data
            best_data_train = data_train


        if train_avg_loss.data.item() < options.estop:
            break


    data_test = test(model, test_loader, use_cuda)

    print('best_train')
    reportMetrics(best_data_train, f'./{model_name}_results/{model_name}_train_data.txt', True)
    print('best_valid')
    reportMetrics(best_data_valid, f'./{model_name}_results/{model_name}_valid_data.txt', True)
    print('test')
    reportMetrics(data_test, f'./{model_name}_results/{model_name}_test_data.txt', True)
    train_loss_f.close()
    train_acc_f.close()
    train_pre_f.close()
    train_rec_f.close()
    train_f1_f.close()
    valid_acu_f.close()
    valid_pre_f.close()
    valid_rec_f.close()
    valid_f1_f.close()

def train(model, train_loader, use_cuda, criterion, optimizer):
    # main training loop
    train_loss = 0.0

    y_hat_tesnor = torch.empty(0)
    y_tensor = torch.empty(0)
    model.train()
    for it, train_data in enumerate(train_loader):
        vote = []
        for data_dic in train_data:
            if use_cuda:
                imgs, labels = data_dic['image'].cuda(), data_dic['label'].cuda()
            else:
                imgs, labels = data_dic['image'], data_dic['label']

            integer_encoded = labels.data.cpu().numpy()
            # target should be LongTensor in loss function
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
            if use_cuda:
                ground_truth = ground_truth.cuda()
            train_output = model(imgs)
            _, predict = train_output.topk(1)
            loss = criterion(train_output, ground_truth)
            train_loss += loss
            vote.append(predict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        vote = torch.cat(vote, 1)
        final_vote, _ = torch.mode(vote, 1)
        ground_truth = train_data[0]['label']
        y_hat_tesnor=torch.cat((y_hat_tesnor,final_vote.cpu().data))
        y_tensor=torch.cat((y_tensor, ground_truth))

    data = getData(y_hat_tesnor.numpy(), y_tensor.numpy())

    return train_loss, data

def validate(model, valid_loader, use_cuda):
    y_hat_tesnor = torch.empty(0)
    y_tensor = torch.empty(0)
    model.eval()
    for it, valid_data in enumerate(valid_loader):
        vote = []
        for data_dic in valid_data:
            if use_cuda:
                imgs, labels = data_dic['image'].cuda(), data_dic['label'].cuda()
            else:
                imgs, labels = data_dic['image'], data_dic['label']
            valid_output = model(imgs)
            _, predict = valid_output.topk(1)
            vote.append(predict)

        vote = torch.cat(vote, 1)
        final_vote, _ = torch.mode(vote, 1)
        ground_truth = valid_data[0]['label']
        y_hat_tesnor=torch.cat((y_hat_tesnor,final_vote.cpu().data))
        y_tensor=torch.cat((y_tensor, ground_truth))
    
    data = getData(y_hat_tesnor.numpy(), y_tensor.numpy())
    return data

def test(model, test_loader, use_cuda):
    y_hat_tesnor = torch.empty(0)
    y_tensor = torch.empty(0)
    model.eval()
    for it, test_data in enumerate(test_loader):
        vote = []
        for data_dic in test_data:
            if use_cuda:
                imgs, labels = data_dic['image'].cuda(), data_dic['label'].cuda()
            else:
                imgs, labels = data_dic['image'], data_dic['label']
            test_output = model(imgs)
            _, predict = test_output.topk(1)
            vote.append(predict)

        vote = torch.cat(vote, 1)
        final_vote, _ = torch.mode(vote, 1)
        ground_truth = test_data[0]['label']
        y_hat_tesnor=torch.cat((y_hat_tesnor,final_vote.cpu().data))
        y_tensor=torch.cat((y_tensor, ground_truth))
    
    data = getData(y_hat_tesnor.numpy(), y_tensor.numpy())
    return data

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)