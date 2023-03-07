import matplotlib.pyplot as plt

average_loss = []
average_test_acc = []
epochs = range(100)
with open ('test_accuracy.txt') as f:
  lines = f.readlines()
  for line in lines:
    average_test_acc.append(float(line))

## change count condition literal and division if batch/epoch # change

with open ('train_loss.txt') as f:
  lines = f.readlines()
  temp_loss = []
  count = 0
  for line in lines:
    count += 1
    temp_loss.append(float(line))
    if count == 75:
      average_loss.append(sum(temp_loss)/75)
      count = 0
      temp_loss = []

print(f'max avg test acc: {max(average_test_acc)} at epoch: {average_test_acc.index(max(average_test_acc))}')
plt.xlabel('epochs')
plt.ylim(0, 1)
plt.plot(epochs, average_loss, label = "average loss")
plt.plot(epochs, average_test_acc, label = "average test acc")
plt.savefig('AlexNet2D_SE'+str(len(epochs))+'.png')
plt.legend()
plt.show()


