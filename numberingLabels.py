import os
import numpy
import csv
from PIL import Image
from PIL import ImageFile
import tensorflow as tf

f  = open("labels.csv")
f2 = open("label_dict.csv", 'w')
reader = csv.reader(f)
reader.__next__()
labels_dict = {}
idx = 0
for line in reader:
    if line[1] in labels_dict:
        continue
    else:
        labels_dict[line[1]] = idx
        f2.write(line[1] + "," + str(idx) + "\n")
        idx += 1
f.close()
f2.close()
