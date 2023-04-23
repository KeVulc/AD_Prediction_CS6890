import matplotlib.pyplot as plt

## change count condition literal and division if batch/epoch # change


pretrained_loss = []
pretrained_se_loss = []
topology_se_loss = []
topology_loss = []


with open ('./AlexNet2D_Pretrained_results/train_loss.txt') as f:
  lines = f.readlines()
  for line in lines:
    pretrained_loss.append(float(line))



with open ('./AlexNet2D_SE_Pretrained_results/train_loss.txt') as f:
  lines = f.readlines()
  for line in lines:
    pretrained_se_loss.append(float(line))



with open ('./AlexNet2D_SE_Topology_results/train_loss.txt') as f:
  lines = f.readlines()
  for line in lines:
    topology_se_loss.append(float(line))

with open ('./AlexNet2D_Topology_results/train_loss.txt') as f:
  lines = f.readlines()
  for line in lines:
    topology_loss.append(float(line))


plt.xlabel('epochs')
plt.ylim(0, 10)
plt.plot(range(len(pretrained_loss)), pretrained_loss, label = f"AlexNet2D_Pretrained loss")
plt.plot(range(len(pretrained_se_loss)), pretrained_se_loss, label = f"AlexNet2D_SE_Pretrained loss")
plt.plot(range(len(topology_se_loss)), topology_se_loss, label = f"AlexNet2D_Topology loss")
plt.plot(range(len(topology_loss)), topology_loss, label = f"AlexNet2D_Topology loss")
plt.legend(loc='upper left')
plt.savefig(f'./losses.png')
plt.clf()

# import sys
# metrics = sys.argv[1:]
# if 'all' in metrics:
#   metrics = ['acc', 'rec', 'pre', 'f1']
# train_f1 = []
# valid_f1 = []
# for metric in metrics:
#   if metric == 'acc':
#     metric_name = 'accuracy'
#   elif metric == 'rec':
#     metric_name = 'recall'
#   elif metric == 'pre':
#     metric_name = 'precision'
#   elif metric == 'f1':
#     metric_name = 'f1'
#   train_metric = []
#   valid_metric = []
#   with open (f'./valid_{metric_name}.txt') as f:
#     lines = f.readlines()
#     for line in lines:
#       valid_metric.append(float(line))
#   if metric_name == 'f1':
#     train_f1 = [x for x in train_metric]
#     valid_f1 = [x for x in valid_metric]

#   with open (f'./train_{metric_name}.txt') as f:
#     lines = f.readlines()
#     for line in lines:
#       train_metric.append(float(line))

#   model_name = 'AlexNet2D_Pretrained'
#   # model_name = 'AlexNet2D_SE_Pretrained'
#   # model_name = 'AlexNet2D_SE_Topology'
#   # model_name = 'AlexNet2D_Topology'

#   epochs = range(len(train_metric))

#   plt.xlabel('epochs')
#   plt.ylim(0, 1)
#   plt.plot(epochs, train_metric, label = f"train {metric_name}")
#   plt.plot(epochs, valid_metric, label = f"valid {metric_name}")
#   plt.legend(loc='upper left')
#   plt.savefig(f'./{model_name}_results/{model_name}_{len(epochs)}_{metric_name}.png')
#   plt.clf()


# print(f'best model valid f1: {max(valid_f1)} at epoch: {valid_f1.index(max(valid_f1))}')