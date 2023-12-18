import time
import tntp
import networkx as nx
from graph_flow import Graph
import matplotlib as mpl
import matplotlib.pyplot
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


n, edges = tntp.read_network("./TransportationNetworks/Anaheim/Anaheim_net.tntp")
d_flow = tntp.read_trips("./TransportationNetworks/Anaheim/Anaheim_trips.tntp")
d_flow = [(u, v, f*0.51) for (u, v, f) in d_flow if f > 0]

res = {}
failed_sets = []
queue = [56, 122, 211, 412, 537, 541, 546, 845, 862, 863, 871, 904]
d = [0.21, 0.22, 0.23]#[0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.15, 0.18, 0.21, 0.24, 0.31, 0.34, 0.37, 0.41, 0.48, 0.6, 0.7]
times = []
for p in queue:
    start = time.time()
    g = Graph(n, eps=1e-8, edges=edges)
    g.add_cost(175, 0.22)
    g.add_cost(492, 0.06)
    g.add_cost(871, 0.01)
    g.add_cost(641, 0.3)
    g.add_cost(278, 1.)
    g.add_cost(845, 0.23)
    
    g.set_flow(d_flow)
    done, clc_flow = g.calculate_flow()
    if done:
        braess, costs, sum = g.calculate_time()
    finish = time.time()
    times.append(finish - start)
        #print(braess)
    res[p] = sum
    #    
    #    sets = list(tuple(sorted(s)) for s in powerset(braess))
    #    for set in sets:
    #        if not (set in res):
    #            fail = False
    #            for f_s in failed_sets:
    #                if set(f_s).issubset(f_s):
    #                    fail = True
    #                    break
    #            if not fail:
    #                queue.append(set)
    #else:
    #    failed_sets.append(x)
    #    
print(sorted([(w, i) for (i, w) in res.items()]))
#print("times: ", times)
#sum_time = 0
#for t in times:
#    sum_time += t
#print("avg: ", sum_time/len(times))

    
#G = nx.MultiDiGraph()
#G.add_edges_from([(edge[0], edge[1]) for edge in edges])

#pos = nx.spring_layout(G, seed=63)
#fig = matplotlib.pyplot.figure()
#nx.draw_networkx(G, pos, ax=fig.add_subplot())#, **options)
#mpl.use("Agg") 
#fig.savefig("./plots/graph"+".png")


