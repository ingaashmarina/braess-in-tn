from graph_flow import Graph
import networkx as nx

x = Graph(4, [
    (0, 1, 90, 10),
    (0, 2, 100, 1),
    (1, 3, 110, 3),
    (2, 1, 130, 2),
    (2, 3, 140, 11)
])

x.set_flow(
    [(0, 3, 120)]
)
print(x.calculate_flow())
x.block_edge(x.find_braess()[0])
x.reset_flow()
print(x.calculate_flow())
print(x.find_braess())

print("===============")

x = Graph(3, [
    (0, 1, 300, 1),
    (0, 2, 100, 10),
    (1, 2, 201, 2)
])
x.set_flow(
    [(0, 2, 150),
     (1, 2, 100)]
)
print(x.calculate_flow())
x.block_edge(x.find_braess()[0])
print(x.calculate_flow())
