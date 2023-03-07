import os

S_true = set([x for x in os.walk('./Image')][0][2])
S_missing = set([])
valid_ct = 0
all_files = ['old_test_2C_new.txt', 'old_test_2classes.txt', 'old_test.txt', 'old_train_2C_new.txt', 'old_train_2classes.txt', 'old_train.txt', 'old_validation_2C_new.txt']
for file_name in all_files:
  S_written = set([])
  print()
  print(file_name)
  with open(file_name) as f:
      lines = f.readlines()
      for line in lines:
          sub_file = line.split()[0]
          sub_label = line.split()[1]
          if sub_file not in S_true:
            S_missing.add(sub_file)
            # print(sub_file)
          else:
             valid_ct += 1
             with open('./'+file_name[4:], 'a') as wf:
              if sub_file not in S_written:
                print(sub_file, sub_label, file=wf)
                S_written.add(sub_file)