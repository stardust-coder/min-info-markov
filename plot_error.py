import matplotlib.pyplot as plt
import numpy as np
from numpy import array

result = [array([0.41983871, 0.35848004]), array([0.48348168, 0.43759347]), array([0.55340426, 0.53752056]), array([0.40993801, 0.42578616]), array([0.38902238, 0.4152643 ]), array([0.47526173, 0.36676252]), array([0.4480164 , 0.43114328]), array([0.54792177, 0.33162845]), array([0.59213678, 0.45574026]), array([0.43455206, 0.39623059]), array([0.51688966, 0.47823966]), array([0.54728323, 0.56115875]), array([0.43158232, 0.34168267]), array([0.46056222, 0.45935939]), array([0.4896812 , 0.46812073]), array([0.60565173, 0.38236749]), array([0.47418409, 0.44529312]), array([0.41631474, 0.44558558]), array([0.50938596, 0.46537044]), array([0.48698213, 0.47896588]), array([0.5253738 , 0.41340965]), array([0.46990391, 0.46282748]), array([0.48293642, 0.47328346]), array([0.54274524, 0.3851799 ]), array([0.42305746, 0.43649647]), array([0.49636686, 0.37335196]), array([0.56701669, 0.50305589]), array([0.45792076, 0.4703385 ]), array([0.55347194, 0.45203711]), array([0.56605604, 0.49819508])]
x = []
y = []
z = []
for r in result:
    x.append(r[0][0])
    y.append(r[1][0])
    z.append(r[2][0])

plt.figure(figsize=(5,5))
# plt.scatter(x,y)
# plt.savefig("error_plot.png")

plt.clf()

#plt.hist(x,label="φ=0.5")
plt.hist(x,label="φ1=0.5")
plt.hist(y,label="φ2=0.3")
plt.hist(z,label="φ3=0.1")
plt.legend()
plt.savefig("error_dist.png")