import time
import tntp
import networkx as nx
from graph_flow import Graph
import matplotlib as mpl
import matplotlib.pyplot
from itertools import chain, combinations

n, edges = tntp.read_network("./TransportationNetworks/Anaheim/Anaheim_net.tntp")
#5, [
#    (1, 2, 6, 5000),
#    (2, 3, 100, 1000),
#    (2, 4, 100, 4000),
#    (2, 5, 100, 6000),
#    (3, 4, 5, 2000),
#    (4, 5, 100, 15000),
#    (4, 1, 100, 3000)
#]#tntp.read_network("./TransportationNetworks/Anaheim/Anaheim_net.tntp")
d_flow = tntp.read_trips("./TransportationNetworks/Anaheim/Anaheim_trips.tntp")
#[
#    (1, 4, 3.11),
#    (3, 5, 4.5)]#tntp.read_trips("./TransportationNetworks/Anaheim/Anaheim_trips.tntp")
d_flow = [(u, v, f*0.51) for (u, v, f) in d_flow if f > 0]


g = Graph(n, eps=1e-8, edges=edges)

start = time.time()
g.set_flow(d_flow)
done, clc_flow = g.calculate_flow()
if done:
    calc, braess, costs, sum = g.calculate_time()
    w = time.time() - start
    if calc:
        print(braess)
        #print(costs)
        print(sum)
        print("t: ", w)
    else:
        print("f")
else:
    print("clogged")



