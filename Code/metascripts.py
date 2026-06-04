import numpy as np
import pandas as pd
def read_files(model_name, lr, folds, max_epochs, mode, extra = ""):
    results_std = []
    results_min = []
    results_max = []
    results_avg = []

    print(mode, "\n", "--"*80)
    for fold in range(folds):
        out = []

        with open('./metadata/{}/{}_{}_{}_fold_{}'.format(model_name+extra, mode, model_name, lr, fold + 1)) as f:
            for line in f.readlines():
                line = line.strip()
                out.append(float(line))


        stda = np.std(out)
        average = np.mean(out)
        min = np.min(out)
        max = np.max(out)
        print("{}:: Fold--{} | min--{} | max--{} | average--{} | std--{}".format(mode, fold+1, min, max, average, stda))
        results_std.append(stda)
        results_min.append(min)
        results_max.append(max)
        results_avg.append(average)
    print("--"*80)
    mean_all_folds = np.mean(results_avg)
    std_all_folds = np.mean(results_std)
    max_all_folds = np.mean(results_max)
    print("All Folds Mean {} | All Folds Std {} | All Folds Max {}".format(mean_all_folds*100, std_all_folds*100, max_all_folds))

def read_files_updated(model_name, lr, folds, max_epochs, mode, extra = ""):

    print(mode, "\n", "--"*80)

    values = []

    for fold in range(folds):
        temp = []

        with open('./metadata/{}/{}_{}_{}_fold_{}'.format(model_name+extra, mode, model_name, lr, fold + 1)) as f:
            for line in f.readlines():
                line = line.strip()
                temp.append(float(line))
        values.append(temp)

    average = []
    standard_dev = []
    upper = []
    lower = []
    for epoch in range(max_epochs):
        temp = []
        for fold in range(folds):
            temp.append(values[fold][epoch])
        m = np.mean(temp)
        s = np.std(temp)
        average.append(m)
        standard_dev.append(s)
        upper.append(m+s)
        lower.append(m-s)

    print(average)
    print(mode, "\n", "--" * 80)
    print(standard_dev)
    print(mode, "\n", "--" * 80)
    print(upper)
    print(mode, "\n", "--" * 80)
    print(lower)

    data = {'Mean': average,
            'std': standard_dev,
            'Upper': upper,
            'lower': lower}

    df = pd.DataFrame.from_dict(data)


    df.to_csv('./csvfiles/{}_{}_{}.csv'.format(model_name+extra,lr,mode), index=False)


variant = ["_18", "_34", "_50"]

def helper():
    for i in range(1, 4):
        print("Type {} to get statistics about resnet model variant{}".format(i, variant[i - 1][1:]))
    flag = True
    while (flag):
        n = int(input())
        if (n <= 3 and n >= 1):
            extra = variant[n-1]

            flag = False
        else:
            print("Enter a valid value")

    return extra

modes = ["train_accu", "val_accu", "train_loss", "val_loss", "train_f1", "val_f1",
             "train_confusion_matrix", "val_confusion_matrix", "test_confusion_matrix"]
folds = 10
models = ["alexnet", "resnet_v0", "cnnnet"]
lrs = [1e-4, 1e-3, 1e-2]
lr = 0.0001
model_name = "resnet"
print("Heyyy.............. ")
for i in range(1, 4):
    print("Type {} --------------------->>>>>model {}".format(i, models[i-1]))
flag1 = True
extra = ""
while(flag1):
    n = int(input())
    if(n <= 3 and n >= 1):
        model_name = models[n-1]
        if(model_name == "resnet_v0"):
            extra = helper()
        flag1 = False
    else:
        print("Enter a valid value")

for i in range(1, 4):
    print("Type {} to get select the learning rate {}".format(i, str(lrs[i-1])))
flag2 = True
while(flag2):
    n = int(input())
    if(n <= 3 and n >= 1):
        lr = lrs[n-1]
        flag2 = False
    else:
        print("Enter a valid value")

max_epochs = 50

"""
for mode in modes[:6]:
    read_files(model_name, lr, folds, max_epochs, mode, extra=extra)
"""
final_modes = ["train_accu", "val_accu", "train_loss", "val_loss"]
for mode in final_modes:
    read_files_updated(model_name, lr, folds, max_epochs, mode, extra=extra)

