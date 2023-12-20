from typing import List
from . import graph_flow
import networkx as nx

class Graph(graph_flow.Graph):
    def __init__ (self, n: int, edges: List[tuple], eps: float = 1e-9):
        super().__init__(n, eps, edges)
        
        self.G = nx.MultiDiGraph()
        self.G.add_edges_from([(edge[0], edge[1]) for edge in edges])
        