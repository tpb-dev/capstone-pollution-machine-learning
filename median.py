# -*- coding: utf-8 -*-
import csv
import sys
import numpy
import pandas
import matplotlib.pyplot as plt
import collections


f = open("prsa.csv", "rb")
reader = csv.reader(f)

i = 0

dic = collections.defaultdict(list)

for row in reader:
    if i == 0:
        i += 1
        continue
    bad = False
    for i in row:
        if i == 'NA':
            bad = True
            break
    if not bad:
        dic[int(row[4])].append(float(row[5]))
f.close()

for key,value in dic.items():
    dic[key] = numpy.median(value)

D = dic

plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.xlabel("Hour Median")
plt.ylabel("PM 2.5 level")
plt.title("PM 2.5 level vs. Hour Median")

plt.show()
