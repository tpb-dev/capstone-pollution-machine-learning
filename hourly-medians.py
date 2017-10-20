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

#print "Length of arrray: ", len(arr)

arr = np.array(arr)


#print arr


"""
pl.plot(arr[:,1], arr[:,5])
pl.xlabel("Time")
pl.ylabel("PM 2.5 value")
pl.title("PM 2.5 Value vs Time")
pl.show()
"""

nami = arr[:,5].astype(np.float)

h = np.sort(nami)

fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

plt.xlabel("PM 2.5 values")
plt.ylabel("Density of Probability")
plt.title("PM 2.5 Values Normal Distribution")

plt.plot(h,fit,'-o')

plt.hist(h,normed=True)      #use this to draw histogram of your data

plt.show()                   #use may also need add this





"""
#objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
#objects = arr[:,1]

#y_pos = np.arange(len(objects))
#performance = [10,8,6,4,2,1]
#performance = arr[:,5]

#plt.bar(y_pos, performance, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('Usage')
#plt.title('Programming language usage')

#plt.show()
"""

"""

h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
     187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
     161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted

fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

pl.plot(h,fit,'-o')

pl.xlabel("Time")
pl.ylabel("PM 2.5 value")
pl.title("PM 2.5 Value vs Time")

pl.hist(h,normed=True)      #use this to draw histogram of your data

pl.show()                   #use may also need add this

"""
