from enum import Enum
import networkx as nx

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

import numpy as np
import matplotlib.pyplot as plt

class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2

def count_infected(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.INFECTED])

def count_susceptible(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.SUSCEPTIBLE])

def count_removed(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.REMOVED])

class Population(Model):
    """Population"""

    def __init__(self, population_size, initial_outbreak_size, spread_chance):
        print("Beginning model setup...\n")
        self.population_size = population_size
        print("Creating graph...")
        self.graph = nx.powerlaw_cluster_graph(population_size, 100, 0.5)
        #self.graph = nx.complete_graph(population_size)
        print(len(self.graph.edges))
        print("Initializing grid...")
        self.grid = NetworkGrid(self.graph)
        self.schedule = SimultaneousActivation(self)
        self.initial_outbreak_size = initial_outbreak_size
        self.spread_chance = spread_chance

        print("Initializing data collector...")
        self.datacollector = DataCollector({"Infected:" : count_infected,
                                            "Susceptible:" : count_susceptible,
                                            "Removed:" : count_removed})

        for i, node in enumerate(self.graph.nodes()):
            a = Person(i, self, State.SUSCEPTIBLE, spread_chance)
            self.schedule.add(a)
            self.grid.place_agent(a, i)
            if i % 100 == 0:
                print("Finished with agent ", i)

        infected_nodes = self.random.sample(self.graph.nodes(), self.initial_outbreak_size)
        for a in self.grid.get_cell_list_contents(infected_nodes):
            a.status = State.INFECTED

        self.datacollector.collect(self)
        print("Model initialized...\n")

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def run(self, n):
        for i in range(n):
            print("Steps completed: ", i)
            self.step()



class Person(Agent):
    def __init__(self, unique_id, model, status, spread_chance):
        super().__init__(unique_id, model)
        self.status = status

        self.spread_chance = spread_chance
        self.to_be_infected = []
        self.time_to_recovery = np.random.poisson(5)

    def get_neighbors_to_infect(self):
        neighbor_ids = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbor_ids)]
        for n in neighbors:
            if n.status == State.SUSCEPTIBLE:
                if self.random.random() < self.spread_chance:
                    self.to_be_infected.append(n)

    def infect_neighbors(self):
        for a in self.to_be_infected:
            a.status = State.INFECTED
        self.to_be_infected = []

    def step(self):
        if self.status == State.INFECTED:
            self.get_neighbors_to_infect()

    def advance(self):
        self.infect_neighbors()
        if self.status == State.INFECTED:
            self.time_to_recovery -= 1
        if self.time_to_recovery == 0:
            self.recover()

    def recover(self):
        self.status = State.REMOVED


#model = Population(1000, 1, 0.001)
#model = Population(1000, 1, 0.005)
model = Population(1000, 10, 0.005)
model.run(25)
df = model.datacollector.get_model_vars_dataframe()
print(model.graph.edges)
#print(df.head())
#df.plot()
'''
nx.draw(model.graph)
options = {"node_size": 10, "alpha": 0.8}
pos = nx.spring_layout(model.graph)
nx.draw_networkx_nodes(model.graph, pos, nodelist=list(model.graph.nodes), **options)
nx.draw_networkx_edges(model.graph, pos, width=0.1, alpha=0.5)
nx.draw_networkx_edges(
    model.graph,
    pos,
    edgelist=list(model.graph.edges),
    width=0.1,
    alpha=0.5,
    edge_color='r'
)
'''
plt.axis("off")
plt.show()
