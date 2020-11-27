import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from statistics import mean, median,variance,stdev




G_practice = nx.DiGraph()

weighted_edges = [[0,1,0.1],
                  [0,2,0.2],
                  [1,3,0.3],
                  [1,4,0.4],
                  [2,4,0.5],
                  [2,0,0.6],
                  [3,0,0.7],
                  [3,1,0.8],
                  [4,3,0.9],
                  [4,2,1.0]
                 ]

G_practice.add_weighted_edges_from(weighted_edges)


#intVを特徴量としてそのままいれた場合の相関係数
data = [0.1032442932349916, -0.2732497168556151, 0.337244097390134, -0.16822175738515896, -0.08088116463039255, -0.03584148765447488, 0.03489194226059807, -0.24320860967893695, 0.13655351293388573, -0.18537858789119677, -0.18367239539007288, 0.2541472376194171, 0.1747112371902146, 0.11577733368017398, -0.21919262823684574, -0.19285381078232258, -0.0020717068558618857, -0.2853968476945913, -0.04822308586420633, -0.10903483137492004, -0.3458346778939076, 
-0.16812895748439705, 0.029896520008611687, -0.1091987279217082, -0.2097918916441572, -0.28151863382937725, -0.1734179236616784, -0.057081656910616396, -0.33991150963240885, 0.17336662396973093, -0.2345226584114239, 0.14214948772055824, 0.1618545067498431, -0.4168423420950616, -0.18919311371111305, -0.21753703881403577, -0.3605376235945272, -0.14135356471235724, -0.09639854906423685, -0.08247547837743455, -0.15914690789686284, -0.2563885069033204, -0.1470414726966376, -0.18999264538639676, 0.01879589765882556, 0.079129919523166, 0.060889133935099427, -0.27748578070019203, 0.257894722488139, -0.12625219659129672]

#intVをsoftmaxによって正規化したときの相関係数
data2 = [-0.647768503442612, -0.4538575534296588, -0.4902348385345789, -0.3635577546711713, -0.4329726841758543, -0.549500362842788, -0.4870982272459474, -0.4714182141231938, -0.5876239275248574, -0.5260033156702739, -0.38397555538476114, -0.44862518212484026, -0.5300250730942386, -0.32061311681424454, -0.5174724501959507, -0.4227914201380135, -0.5468792782194097, -0.5141759512282825, -0.46207987056041316, -0.44784161074427836, -0.5471070484608549, -0.4279946355581717, -0.5167432928819723, -0.4291397500977234, -0.3140756441079165, -0.3867628774935683, -0.33404405946174875, -0.3857441516225389, -0.5277464142689426, -0.5372019961184358, -0.49372347216763, -0.5358622929998879, 
-0.6131228751496627, -0.6700138873213024, -0.6145134548313071, -0.4336462147911721, -0.5279759344565287, -0.4107896746824566, -0.4685346137584172, -0.485678054698642, -0.650641510162316, -0.5259106888285607, -0.35983937549469025, -0.48509187608735826, -0.47027163084309775, -0.5264938177748961, -0.4450053265082373, -0.44401183095120766, -0.5374212858441549, -0.5721884543323786]



m = mean(data2)
median = median(data2)
variance = variance(data2)
stdev = stdev(data2)
print('平均: {0:.4f}'.format(m))
print('中央値: {0:.4f}'.format(median))
print('分散: {0:.4f}'.format(variance))
print('標準偏差: {0:.4f}'.format(stdev))



