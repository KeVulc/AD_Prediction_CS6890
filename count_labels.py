import os
filenames = ['split_train', 'split_validation', 'split_test',]

ad_count = 0
norm_count = 0
for filename in filenames:
    with open(f'./{filename}.txt') as f:
        for line in f.readlines():
            if "Normal" in line:
                ad_count+=1
            else:
                norm_count+=1
            
            
print(f'AD:{ad_count} Normal:{norm_count} total:{ad_count+norm_count}')