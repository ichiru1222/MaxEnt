import networkx as nx
import math
import random
import numpy as np
import matplotlib.pyplot as plt

n = 40
p = 0.5

class Graphenv:

 

    def __init__(self, graph, reward, random_error = None):
        #e.g.) shape = [4,4] , reward = np.zeos(nS)

        
        self.nS = nx.number_of_nodes(graph) #状態数　（ノードの数）
        #print(self.nS)
        self.nA = [x[1] for x in graph.degree()] #行動数　各状態に対しての行動数をリストに格納 [2,3,1,9]各状態に対する次数
        self.graph = graph
        self.P = self.make_prob()


    def make_prob(self):
        """
        (ex:output) 横：状態数　縦：最大行動数 pは確定的
                    [[[0. 0. 1. 0. 0.]
                    [0. 0. 0. 1. 0.]
                    [0. 0. 0. 0. 0.]]

                    [[0. 0. 1. 0. 0.]
                    [0. 0. 0. 0. 1.]
                    [0. 0. 0. 0. 0.]]

                    [[1. 0. 0. 0. 0.]
                    [0. 1. 0. 0. 0.]
                    [0. 0. 0. 1. 0.]]

                    [[1. 0. 0. 0. 0.]
                    [0. 0. 1. 0. 0.]
                    [0. 0. 0. 0. 1.]]

                    [[0. 1. 0. 0. 0.]
                    [0. 0. 0. 1. 0.]
                    [0. 0. 0. 0. 0.]]]
        """
        Alen = max([x[1] for x in self.graph.degree()])
        P = np.array([[list(np.identity(self.nS)[i]) for i in [x[1] for x in self.graph.edges([e])]] + [[0] * self.nS] * (Alen - len(self.graph.edges([e]))) for e in range(self.nS)], dtype=float)

        return P



def make_random_graph(number_of_nodes, p): #number_of_nodes:ノードの数，p:ノード間にエッジを生成する確率
    while True: #グラフを連続にする
        graph = nx.fast_gnp_random_graph(number_of_nodes, p)
        if nx.is_connected(graph):
            break
    return graph


def make_expart_paths(graph, number_of_exparts):  #エキスパート軌跡の生成 maxentは軌跡の状態数をそろえる必要がある
    len_path = dict(nx.all_pairs_dijkstra(graph))
    all_shortest_paths = {} #あるノードから別のノードまでの最短経路を格納　{0:{0:[0],1:[0,1]},1:{2:[1,3,2]}} 辞書の中に辞書
    for i in range(nx.number_of_nodes(graph)):
        all_shortest_paths[i] = len_path[i][1]
    
    #ノードをランダムに二つ選択し，それらの最短経路を取り出す
    exparts_paths = []
    for i in range(number_of_exparts):
        first_node = random.randint(0, nx.number_of_nodes(graph)-1)
        while True:
            second_node = random.randint(0, nx.number_of_nodes(graph)-1)
            if second_node != first_node:
                break
        exparts_paths.append(all_shortest_paths[first_node][second_node])
    
    len_path_list = [len(x) for x in exparts_paths]
    max_path = max(len_path_list)

    exparts_paths_eq = [] #状態数をそろえた軌跡を格納

    for path in exparts_paths:
        if len(path) != max_path:
            for i in range(max_path - len(path)):
                path.append(path[-1])
            exparts_paths_eq.append(path)
        else:
            exparts_paths_eq.append(path)

    return exparts_paths_eq

def make_random_goal_fixed_path(graph, number_of_exparts, goal_node):  #エキスパート軌跡の生成 maxentは軌跡の状態数をそろえる必要がある goal_node :ゴールとなるのーどの番号を指定 "random"を入力したらランダムにゴールを指定（エピソードごとにゴールとなるのーどが異なるようになる）
    len_path = dict(nx.all_pairs_dijkstra(graph))
    all_shortest_paths = {} #あるノードから別のノードまでの最短経路を格納　{0:{0:[0],1:[0,1]},1:{2:[1,3,2]}} 辞書の中に辞書
    for i in range(nx.number_of_nodes(graph)):
        all_shortest_paths[i] = len_path[i][1]
    
    #startをランダムに選択し，ゴールは引数からとる．スタートとゴールが同一な軌跡は考えない．それらの最短経路を取り出す
    exparts_paths = []
    gn = goal_node
    if goal_node == "random":
        gn = random.randint(0, nx.number_of_nodes(graph)-1)
    for i in range(number_of_exparts):

        start_node = random.randint(0, nx.number_of_nodes(graph)-1)
        while gn == start_node:
            start_node = random.randint(0, nx.number_of_nodes(graph)-1)
        
        
        
        exparts_paths.append(all_shortest_paths[start_node][gn])
    
    len_path_list = [len(x) for x in exparts_paths]
    max_path = max(len_path_list)

    exparts_paths_eq = [] #状態数をそろえた軌跡を格納

    for path in exparts_paths:
        if len(path) != max_path:
            for i in range(max_path - len(path)):
                path.append(path[-1])
            exparts_paths_eq.append(path)
        else:
            exparts_paths_eq.append(path)

    return exparts_paths_eq


def make_one_expart_paths(graph, number_of_exparts): #一つの軌跡を作成 0から39に至る最短経路
    path_list = list(nx.all_simple_paths(graph, target=nx.number_of_nodes(graph)-1, source=0))
    path_max = max(path_list)
    one_exparts_paths = [path_max for i in range(number_of_exparts)]
    return one_exparts_paths
    

def spacesyntax(graph): #各ノードのintVを導出し，
    closeness=nx.closeness_centrality(graph) #TDの導出、intVS:numpy
    mds = {}
    ras = {}
    intVs = np.array([])
    nodes_len = len(list(graph.nodes))
    #print(closeness)
    for key, value in closeness.items():
        mds[key] = 1/value
    #print(mds)
    for key, md in mds.items():
        ras[key] = 2 * (md - 1)/(nodes_len - 2)
    #print(ras)
    dnup = 2 * (nodes_len * (math.log2((nodes_len + 2) / 3) - 1) + 1)
    dndown = (nodes_len - 1) * (nodes_len - 2)
    dn = dnup/dndown
    #print(dn)
    for key, ra in ras.items():
        intVs = np.append(intVs,dn/ra)
    #print(intVs)

    return intVs

def path_relative_frequency(path, number_of_nodes):#軌跡の通った回数を各ノードごとに相対度数として表示
       #path:[[1,2,6,4,4,4],[5,6,7,8,9,10]]の形, number_of_nodes:ノードの数
    frequency = np.zeros(number_of_nodes)
    all_state = 0
    for p in path:
        p_set = set(p)
        all_state += len(p_set)
        for s in p_set:
            frequency[s] += 1

    relative_frequency = frequency/all_state
    return relative_frequency

"""
    for p in path:
        all_state += len(p)
        for s in p:
            frequency[s] += 1
    relative_frequency = frequency/all_state
    return relative_frequency



"""



def graph_view_1st(graph, size, number_of_nodes, ylb):
    #ランダム軌跡を与えたときのぐらふを表示
    #inintV
    #notinintV
    sizes = (size -size.min()) / (size.max() - size.min()) * 1000
 

    # 各頂点に対して円周上の座標を割り当てる
    pos = nx.circular_layout(graph)
    x = [i for i in range(number_of_nodes)]
    y = size

    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(1,2,1,  xlabel='node', ylabel=ylb)
    ax1.bar(x, y)
    ax2 = fig.add_subplot(1,2,2)
    nx.draw(graph, with_labels=True, node_size = sizes, pos=pos)
    plt.show(graph)


def graph_view_2nd(graph, path, reward, number_of_nodes, ylb):
    #同一な軌跡を与えたときのぐらふを表示
    avarage_reward = sum(reward) / number_of_nodes
    #inintV
    #sizes = reward * 300 / avarage_reward
    #notinintV
    sizes = (reward - reward.min()) / (reward.max() - reward.min()) * 1000
    colors = ["green" for i in range(number_of_nodes)]
    for i in path[0]:
        colors[i] = "red"

    # 各頂点に対して円周上の座標を割り当てる
    pos = nx.circular_layout(graph)
    x = [i for i in range(number_of_nodes)]
    y = reward

    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(1,2,1,  xlabel='node', ylabel=ylb)
    ax1.bar(x, y, color=colors)
    ax2 = fig.add_subplot(1,2,2)
    nx.draw(graph, with_labels=True, node_color = colors, node_size = sizes, pos=pos)
    plt.show(graph)


                

                




   



    
    
    








if __name__ == '__main__':
    import shelve


    number_of_nodes = 40
    p = 0.05
    number_of_exparts = 10
    reward = np.zeros(10)
    graph = make_random_graph(number_of_nodes, p)

    graph_data = shelve.open("database")
    #graph_data["graph_1"] = graph
    
    path = make_expart_paths(graph,number_of_exparts)

    #print(path_relative_frequency(path, number_of_nodes))
    #print(sum(path_relative_frequency(path, number_of_nodes)))

    def softmax(a):
        a_max = max(a)
        x = np.exp(a-a_max)
        u = np.sum(x)
        return x/u

    env = Graphenv(graph_data["graph_1"], reward)
    #print(env.P)
    print(env.P.shape)
    print(env.P.dtype)
    print(env.nA)

    print()
    print(spacesyntax(graph_data["graph_1"]))
    print(softmax(spacesyntax(graph_data["graph_1"])))
    print(sum(softmax(spacesyntax(graph_data["graph_1"]))))

    intVs = spacesyntax(graph_data["graph_1"])

    
    for i in range(number_of_nodes):
        graph_data["graph_1"].add_node(i, intV=intVs[i])
    
    print(graph_data["graph_1"].nodes(data=True))
    print(make_one_expart_paths(graph_data["graph_1"], 10))
    print("#######################################################")
    print(make_random_goal_fixed_path(graph_data["graph_1"], 10, "random"))
   

    #print(graph.)

    #print(graph)
    #nx.draw(graph_data["graph_1"], with_labels=True)

    #plt.show(graph_data["graph_1"])