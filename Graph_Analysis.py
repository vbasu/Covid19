import networkx as nx

graph = nx.powerlaw_cluster_graph(1000, 100, 0.01)

L = nx.laplacian_matrix(graph)

print(sum(L[0]))
