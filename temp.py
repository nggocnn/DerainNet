import os
import shutil
path = "D:\Things\PyProject\DerainNet\data\\test\\norain"
for file in os.listdir(path):
    for i in range(1, 15):
        dest = path + "/" + file.split('.')[0] + "_" + str(i) + "." + file.split('.')[-1]
        shutil.copyfile(path + "/" + file, dest)
        print(dest)
    os.remove(path + "/" + file)