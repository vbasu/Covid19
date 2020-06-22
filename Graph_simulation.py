import networkx as nx
import matplotlib.pyplot as plt

class Graph():
    def __init__(self, graph: nx.Graph, individual_types: list):
        self.graph_size = graph.number_of_nodes()
        self.initialize(individual_types)
        self.graph = graph
        self.update_timeseries()

    def initialize(self, individual_types: list):
        pass

    def update_timeseries(self):
        pass
