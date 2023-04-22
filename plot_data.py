import matplotlib.pyplot as plt

## change count condition literal and division if batch/epoch # change

# with open ('./train_loss.txt') as f:
#   lines = f.readlines()
#   temp_loss = []
#   count = 0
#   for line in lines:
#     count += 1
#     temp_loss.append(float(line))
#     if count == 75:
#       average_loss.append(sum(temp_loss)/75)
#       count = 0
#       temp_loss = []

import sys
metrics = sys.argv[1:]
if 'all' in metrics:
  metrics = ['acc', 'rec', 'pre', 'f1']
train_f1 = []
test_f1 = []
epochs = range(100)
for metric in metrics:
  if metric == 'acc':
    metric_name = 'accuracy'
  elif metric == 'rec':
    metric_name = 'recall'
  elif metric == 'pre':
    metric_name = 'precision'
  elif metric == 'f1':
    metric_name = 'f1'
  train_metric = []
  test_metric = []
  with open (f'./test_{metric_name}.txt') as f:
    lines = f.readlines()
    for line in lines:
      test_metric.append(float(line))
  if metric_name == 'f1':
    train_f1 = [x for x in train_metric]
    test_f1 = [x for x in test_metric]

  with open (f'./train_{metric_name}.txt') as f:
    lines = f.readlines()
    for line in lines:
      train_metric.append(float(line))

  model_name = 'AlexNet2D_SE_Topology'
  # model_name = 'AlexNet2D_SE_Pretrained'
  # model_name = 'AlexNet2D_Pretrained'

  plt.xlabel('epochs')
  plt.ylim(0, 1)
  plt.plot(epochs, train_metric, label = f"train {metric_name}")
  plt.plot(epochs, test_metric, label = f"test {metric_name}")
  plt.legend(loc='upper left')
  plt.savefig(f'./{model_name}_results/{model_name}_{len(epochs)}_{metric_name}.png')
  plt.clf()


print(f'best model test f1: {max(test_f1)} at epoch: {test_f1.index(max(test_f1))}')