import numpy
import matplotlib.pyplot as plt

f = list(numpy.loadtxt("features.txt", dtype='str'))
s = list(numpy.loadtxt("scores.txt", dtype='float'))

s, f = zip(*sorted(zip(s,f), reverse=True))

where = list(numpy.where(numpy.array(s) > 0.6)[0])

s = numpy.array(s)
f = numpy.array(f)

s_data = s[where]
f_data = f[where]

index = numpy.arange(s_data.shape[0]) + 0.3

plt.figure(figsize=(20,8))
plt.plot(index, s_data, marker="o", color="dodgerblue")
plt.xlabel("feature", size=18)
plt.xlim(-1,s_data.shape[0]+1)
plt.ylim(0.6,0.95)
plt.ylabel("AUC", size=18)
plt.xticks(index, labels=f_data, rotation=90, fontsize='small')
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("cat15_features.pdf")
