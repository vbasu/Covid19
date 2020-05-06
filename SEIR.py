import logging
from enum import Enum

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class State(Enum):
    SUSCEPTIBLE = 0
    EXPOSED = 1
    ASYMPTOMATIC = 2
    SYMPTOMATIC = 3
    REMOVED = 4


def count_exposed(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.EXPOSED])

def count_asymptomatic(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.ASYMPTOMATIC])

def count_symptomatic(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.SYMPTOMATIC])

def count_susceptible(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.SUSCEPTIBLE])


def count_removed(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.REMOVED])

class Population(Model):
    """Population
    Adapted from https://www.medrxiv.org/content/10.1101/2020.03.18.20037994v1.full.pdf

    Model Parameters:
    spread_chance: probability of infection based on contact
    gamma: mean incubation period
    alpha: probability of become asymptomatic vs symptomatic
    gamma_AR: infectious period for asymptomatic people
    gamma_YR: infectious period for symptomatic people
    delta: death rate due to disease
    """

    def __init__(self, graph, model_parameters):

        # Model initialization
        self.population_size = model_parameters['population_size']
        self.initial_outbreak_size = model_parameters['initial_outbreak_size']
        self.graph = graph
        # self.graph = nx.complete_graph(population_size)
        self.grid = NetworkGrid(self.graph)
        self.schedule = SimultaneousActivation(self)

        self.datacollector = DataCollector({"Exposed": count_exposed,
                                            "Susceptible": count_susceptible,
                                            "Removed": count_removed,
                                            "Asymptomatic": count_asymptomatic,
                                            "Symptomatic": count_symptomatic
                                            })
        self.model_parameters = model_parameters

        for i, node in enumerate(self.graph.nodes()):
            a = Person(i, self, State.SUSCEPTIBLE, model_parameters)
            self.schedule.add(a)
            self.grid.place_agent(a, i)
            if i % 100 == 0:
                print("Finished with agent ", i)

        infected_nodes = self.random.sample(self.graph.nodes(), self.initial_outbreak_size)
        for a in self.grid.get_cell_list_contents(infected_nodes):
            a.status = State.EXPOSED

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
    def __init__(self, unique_id, model, status, parameters):
        super().__init__(unique_id, model)
        self.status = status

        self.spread_chance = parameters['spread_chance']
        self.p_EAY = parameters['EAY']
        self.p_AR = parameters['AR']
        self.p_YR = parameters['YR']
        self.alpha = parameters['alpha']

        self.to_be_infected = []

    def get_neighbors_to_infect(self):
        neighbor_ids = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbor_ids)]
        # If the agent is symptomatic, cut down on their interactions
        if self.status == State.SYMPTOMATIC:
            neighbors = np.random.choice(neighbors, len(neighbors)//2, replace=False)
        for n in neighbors:
            if n.status == State.SUSCEPTIBLE:
                if self.random.random() < self.spread_chance:
                    self.to_be_infected.append(n)

    def infect_neighbors(self):
        for a in self.to_be_infected:
            a.status = State.EXPOSED
        self.to_be_infected = []

    def e_to_ay(self):
        if self.random.random() < self.p_EAY:
            if self.random.random() < self.alpha:
                self.status = State.ASYMPTOMATIC
            else:
                self.status = State.SYMPTOMATIC

    def ay_to_r(self):
        if self.status == State.ASYMPTOMATIC:
            if self.random.random() < self.p_AR:
                self.status = State.REMOVED
        elif self.status == State.SYMPTOMATIC:
            if self.random.random() < self.p_YR:
                self.status = State.REMOVED

    def step(self):
        if self.status in [State.SYMPTOMATIC, State.ASYMPTOMATIC]:
            self.get_neighbors_to_infect()

    def advance(self):
        self.infect_neighbors()
        if self.status == State.EXPOSED:
            self.e_to_ay()
        elif self.status in [State.SYMPTOMATIC, State.ASYMPTOMATIC]:
            self.ay_to_r()


model_parameters = {
    'population_size': 1000,
    'initial_outbreak_size': 10,
    'alpha': 0.7,
    'spread_chance': 0.005,
    'EAY': 1/5,
    'AR': 1/5,
    'YR': 1/5,
}
#graph = nx.powerlaw_cluster_graph(model_parameters['population_size'], 100, 0.5)
graph = nx.complete_graph(model_parameters['population_size'])
model = Population(graph, model_parameters)
model.run(60)
df = model.datacollector.get_model_vars_dataframe()
print(df.head())
df.plot()
#plt.title("Power Law Distribution")
plt.title("Complete Graph")
plt.grid(b=True, which='major')
plt.minorticks_on()
plt.show()
