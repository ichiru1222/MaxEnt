import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from statistics import mean, median,variance,stdev

#ランダム軌跡に対する相関
###notinintV
correlation_freq_reward = [-0.50273784623445, -0.6144187769711743, -0.6703866996231242, -0.5930142465887213, -0.5241556008516465, -0.6455147002693015, -0.6898144822480738, -0.4466456334237511, -0.7668570040018804, -0.6408559312077019, -0.5441498677581511, -0.609915508311943, -0.576792807456011, -0.5703446090798742, -0.5509620465487445, -0.512328608278347, -0.5812632566935242, -0.42199396531979877, -0.5233239717012749, -0.6133222333998859]

correlation_intV_reward = [-0.5305301362475773, -0.5945627543814963, -0.7551497016289379, -0.6723742651174953, -0.5457185641883902, -0.7565945025046837, -0.6519231820022672, -0.5883578504658048, -0.6437758405794886, -0.7412838456099805, -0.7173452056174933, -0.6074586666515698, -0.6304168572019596, -0.7292582593262221, -0.6405643418132747, -0.6565680593159857, -0.7088383597119005, -0.7572354566885812, -0.6587508662893844, -0.580024857348257]

correlation_freq_intV = [0.6635663097075051, 0.6631810990458257, 0.7485605224633111, 0.5590482599880752, 0.66238002551737, 0.6407077419119127, 0.6765011855146696, 0.572922458716377, 0.7640679058183213, 0.7012878015912036, 0.6469368182915523, 0.5554668705064009, 0.656422436011788, 0.6552149965523998, 0.6363894807758705, 0.6984230065975844, 0.5529346569048439, 0.500159409388245, 0.6723574510989434, 0.6844429863975133]

###inintV
correlation_freq_reward_inintV = [-0.3415676357908176, -0.2624343068458465, -0.22706103661778618, -0.24569587574008822, -0.21269145565742162, -0.4066656490113197, -0.40916899568302795, -0.22836498086335774, -0.41492733861891484, -0.4599677685662843, -0.287230007888985, -0.19963168474090537, -0.514614082183596, -0.3688873491032118, -0.4600934603757326, -0.46444686791735085, -0.33343046239100055, -0.3391697783546618, -0.5832864468243295, -0.2643256592235951]

correlation_intV_reward_inintV = [-0.35430080825283355, -0.2755200137195482, -0.2717870278255292, -0.25221679758229143, -0.2587901913110665, -0.25539197130308594, -0.415751768918149, -0.2516407441883238, -0.38560018577965127, -0.40430140038917534, -0.30319006245856167, -0.2589977936529526, -0.5178935325352706, -0.33275200173804903, -0.4613753954240937, -0.4595198816038571, -0.2826162910531592, -0.35493480064691846, -0.5282103071544779, -0.3481619810340069]

correlation_freq_intV_inintV = [0.6965263904889955, 0.6178795294712561, 0.5711350047289105, 0.6611820054490292, 0.5719340313252033, 0.5944146578224301, 0.6436912503053203, 0.5470594651271147, 0.6269862900097359, 0.5746325062450048, 0.5239394828624453, 0.5826421339971071, 0.7551921210929667, 0.65312840987876, 0.6767392845452533, 0.6535331330976176, 0.4749307586874832, 0.749355066380637, 0.7225262446598981, 0.5611266724201739]



print("correlation_freq_reward")
print(type(correlation_freq_reward))
m = mean(correlation_freq_reward)
medi = median(correlation_freq_reward)
var = variance(correlation_freq_reward)
std = stdev(correlation_freq_reward)
print('平均: {0:.4f}'.format(m))
print('中央値: {0:.4f}'.format(medi))
print('分散: {0:.4f}'.format(var))
print('標準偏差: {0:.4f}'.format(std))

print("correlation_intV_reward")
m = mean(correlation_intV_reward)
medi = median(correlation_intV_reward)
var = variance(correlation_intV_reward)
std = stdev(correlation_intV_reward)
print('平均: {0:.4f}'.format(m))
print('中央値: {0:.4f}'.format(medi))
print('分散: {0:.4f}'.format(var))
print('標準偏差: {0:.4f}'.format(std))


print("correlation_freq_intV")
m = mean(correlation_freq_intV)
medi = median(correlation_freq_intV)
var = variance(correlation_freq_intV)
std = stdev(correlation_freq_intV)
print('平均: {0:.4f}'.format(m))
print('中央値: {0:.4f}'.format(medi))
print('分散: {0:.4f}'.format(var))
print('標準偏差: {0:.4f}'.format(std))


print("#################################################################")

print("correlation_freq_reward_inintV")
m = mean(correlation_freq_reward_inintV)
medi = median(correlation_freq_reward_inintV)
var = variance(correlation_freq_reward_inintV)
std = stdev(correlation_freq_reward_inintV)
print('平均: {0:.4f}'.format(m))
print('中央値: {0:.4f}'.format(medi))
print('分散: {0:.4f}'.format(var))
print('標準偏差: {0:.4f}'.format(std))


print("correlation_intV_reward_inintV")
m = mean(correlation_intV_reward_inintV)
medi = median(correlation_intV_reward_inintV)
var = variance(correlation_intV_reward_inintV)
std = stdev(correlation_intV_reward_inintV)
print('平均: {0:.4f}'.format(m))
print('中央値: {0:.4f}'.format(medi))
print('分散: {0:.4f}'.format(var))
print('標準偏差: {0:.4f}'.format(std))


print("correlation_freq_intV_inintV")
m = mean(correlation_freq_intV_inintV)
medi = median(correlation_freq_intV_inintV)
var = variance(correlation_freq_intV_inintV)
std = stdev(correlation_freq_intV_inintV)
print('平均: {0:.4f}'.format(m))
print('中央値: {0:.4f}'.format(medi))
print('分散: {0:.4f}'.format(var))
print('標準偏差: {0:.4f}'.format(std))











