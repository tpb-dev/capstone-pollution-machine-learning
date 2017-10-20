import numpy as np
import scipy.stats as stats
import pylab as plt

import csv

ifile = open("psa.csv", "rb")
reader = csv.reader(ifile)


arr = []

rownum = 0
header = None
for row in reader:
    # Save header row.
    if rownum ==0:
        header = row
        rownum += 1
        continue
    bad = False
    for col in row:
        if col == 'NA':
            bad = True
    if not bad:
        arr.append(row)
    rownum += 1

ifile.close()

arr = np.array(arr)

nami = arr[:,5].astype(np.float)

h = np.sort(nami)

fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

plt.xlabel("PM 2.5 values")
plt.ylabel("Density of Probability")
plt.title("PM 2.5 Values Normal Distribution")

plt.plot(h,fit,'-o')

plt.hist(h,normed=True)      #use this to draw histogram of your data

plt.show()                   #use may also need add this
