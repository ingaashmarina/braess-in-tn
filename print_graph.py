import graph_flow
import matplotlib.pyplot as pltmpl
import matplotlib.pyplot
import matplotlib as mpl
import networkx as nx
import ipywidgets
import IPython
import tntp

n, edges = tntp.read_network("./TransportationNetworks/Anaheim/Anaheim_net.tntp")
G = nx.MultiDiGraph()
G.add_edges_from([(edge[0], edge[1]) for edge in edges])
H = G.copy()
pos = nx.kamada_kawai_layout(G)#, seed=63)
fig = matplotlib.pyplot.figure(figsize=(50,50))
options = {
        
    }
ax = fig.add_subplot()
nx.draw_networkx(H, pos, ax=ax, edge_color="lightblue", node_color="orange",
                    node_size=1200, width=3, font_size=12)#, **options)
nx.draw_networkx_edges(G, pos, edgelist=[(edges[i][0], edges[i][1]) for i in 
                                         [39, 46, 175, 177, 178, 179, 278, 412, 492, 495, 545, 611, 613, 622, 625, 626, 634, 638, 641, 644, 645, 845]], 
                                        ax=ax, edge_color="red",width=5)#, **options)
mpl.use("Agg") 
fig.savefig("./plots/graph_blank"+".png")