import numpy as np
import json

epsilon = 10**-7
def accuracy(data):
    return (data['TP'] + data['TN']) / data['All']


def error_rate(data):
    return (data['FP'] + data['FN']) / data['All']


def sensitivity(data):
    return data['TP'] / data['P']


def specificity(data):
    return data['TN'] / data['N']


def precision(data):
    denom = data['TP'] + data['FP']
    if denom == 0:
        denom = epsilon
    return data['TP'] / denom


def recall(data):
    denom = data['TP'] + data['FN']
    if denom == 0:
        denom = epsilon
    return data['TP'] / denom


def F1(data):
    denom = precision(data) + recall(data)
    if denom == 0:
        denom = epsilon
    return (2 * precision(data) * recall(data)) / denom


def getConfusion(data):
    return np.array([
        [data['TP'], data['FN'], data['P']],
        [data['FP'], data['TN'], data['N']],
        [data['P`'], data['N`'], data['All']],
    ])


def getData(y_hat, y):
    data = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'P': 0, 'N': 0, 'P`': 0, 'N`': 0, 'All': 0}
    for i in range(len(y)):
        actual = y[i]
        predicted = y_hat[i]
        if actual == 1 and predicted == 1:
            data['TP'] += 1
        elif actual == 0 and predicted == 0:
            data['TN'] += 1
        elif actual == 0 and predicted == 1:
            data['FP'] += 1
        elif actual == 1 and predicted == 0:
            data['FN'] += 1
    data['P`'] = data['TP'] + data['FP']
    data['N`'] = data['TN'] + data['FN']
    data['P'] = data['TP'] + data['FN']
    data['N'] = data['TN'] + data['FP']
    data['All'] = data['P'] + data['N']
    return data


def reportMetrics(data, filename, save=False):
    print(f'Metrics\n\tAccuracy: {round(accuracy(data), 5)}\n\tError Rate: {round(error_rate(data), 5)}\n\tSensitivity: {round(sensitivity(data), 5)}\n\tSpecificity: {round(specificity(data), 5)}\n\tPrecision: {round(precision(data), 5)}\n\tRecall: {round(recall(data), 5)}\n\tF1: {round(F1(data), 5)}')
    print('Confusion Matrix')
    print(getConfusion(data))
    if save:
        with open(f"{filename}", "w") as outfile:
            json.dump(data, outfile)


if __name__ == '__main__':
    data_raw = {'TP': 6954, 'FP': 412, 'TN': 2588, 'FN': 46, 'P': 7000, 'N': 3000, 'P`': 7366, 'N`': 2634, 'All': 10000}
    confusion = getConfusion(data_raw)
    print(confusion)
    print(f'accuracy: {accuracy(data_raw)}')
    print(f'error rate: {error_rate(data_raw)}')
    print(f'sensitivity: {sensitivity(data_raw)}')
    print(f'specificity: {specificity(data_raw)}')
    print(f'precision: {precision(data_raw)}')
    print(f'recall: {recall(data_raw)}')
    print(f'F1: {F1(data_raw)}')
