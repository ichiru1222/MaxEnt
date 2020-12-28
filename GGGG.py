import networkx as nx
import matplotlib.pyplot as plt

# Graphオブジェクトの作成

 
G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_edges_from = ([(1, 2), (2, 3)])
 
# ネットワークの可視化
nx.draw(G)
plt.show()