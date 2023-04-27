import os
import random

SEED = 11
total = 503
train_length = 336
valid_length = 84
test_length = 83

random.seed(SEED)

assignments_train = [1 for _ in range(train_length)]
assignments_valid = [2 for _ in range(valid_length)]
assignments_test = [3 for _ in range(test_length)]

assignments = assignments_train + assignments_valid + assignments_test
random.shuffle(assignments)
assert len(assignments) == (train_length+valid_length+test_length)

subjects_f = open('subjects.txt', 'r')
target_train_f = open('split_train_model_avg.txt', 'w')
target_valid_f = open('split_validation_model_avg.txt', 'w')
target_test_f = open('split_test_model_avg.txt', 'w')

index = 0
for line in subjects_f.readlines():
    assignment = assignments[index]
    index +=1
    if assignment == 1:
        target_train_f.write(line)
    elif assignment == 2:
        target_valid_f.write(line)
    elif assignment == 3:
        target_test_f.write(line)
    
    
subjects_f.close()
target_train_f.close()
target_valid_f.close()
target_test_f.close()


