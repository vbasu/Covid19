from enum import Enum

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

    The social distancing func takes time passed -> new interaction multiplier
    """

    def __init__(self, graph, model_parameters, social_distancing_func=lambda x: 1):

        # Model initialization
        self.population_size = model_parameters['population size']
        self.initial_outbreak_size = model_parameters['initial outbreak size']
        self.graph = graph
        self.grid = NetworkGrid(self.graph)
        self.schedule = SimultaneousActivation(self)
        self.social_distancing_func = social_distancing_func

        self.datacollector = DataCollector({"Exposed": count_exposed,
                                            "Susceptible": count_susceptible,
                                            "Removed": count_removed,
                                            "Asymptomatic": count_asymptomatic,
                                            "Symptomatic": count_symptomatic
                                            })
        self.model_parameters = model_parameters

        for i, node in enumerate(self.graph.nodes()):
            a = Person(i, self, State.SUSCEPTIBLE, model_parameters, social_distancing_func)
            self.schedule.add(a)
            self.grid.place_agent(a, i)
            if i % 100 == 0:
                logger.info("Finished with agent " + str(i))

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
            logger.info("Steps Completed: " + str(i))
            self.step()

    def cross_influence(self, influence_coefficients, ratios):
        for inf_coeff, ratio in zip(influence_coefficients, ratios):
            susceptibles = list(filter(lambda x: x.status == State.SUSCEPTIBLE, self.grid.get_all_cell_contents()))
            to_flip = self.random.sample(susceptibles, int(inf_coeff*ratio*len(susceptibles)))
            for agent in to_flip:
                agent.status = State.EXPOSED


class Person(Agent):
    def __init__(self, unique_id, model, status, parameters, social_distancing_func):
        super().__init__(unique_id, model)
        self.status = status

        self.spread_chance = parameters['spread_chance']
        self.p_EAY = parameters['EAY']
        self.p_AR = parameters['AR']
        self.p_YR = parameters['YR']
        self.alpha = parameters['alpha']

        self.to_be_infected = []
        self.social_distancing_func = social_distancing_func
        self.days_passed = 0
        self.immunity_period = parameters['immunity period']

    def get_neighbors_to_infect(self):
        neighbor_ids = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbor_ids)]
        # If the agent is symptomatic, cut down on their interactions
        if self.status == State.SYMPTOMATIC:
            affected_neighbors = np.random.choice(neighbors,
                                                  int(len(neighbors)*0.5*self.social_distancing_func(self.days_passed)),
                                                  replace=False)
        elif self.status == State.ASYMPTOMATIC:
            affected_neighbors = np.random.choice(neighbors,
                                                  int(len(neighbors)*self.social_distancing_func(self.days_passed)),
                                                  replace=False)
        for n in affected_neighbors:
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

    def r_to_s(self):
        if self.random.random() < 1/self.immunity_period:
            self.status = State.SUSCEPTIBLE

    def step(self):
        if self.status in [State.SYMPTOMATIC, State.ASYMPTOMATIC]:
            self.get_neighbors_to_infect()

    def advance(self):
        self.infect_neighbors()
        if self.status == State.EXPOSED:
            self.e_to_ay()
        elif self.status in [State.SYMPTOMATIC, State.ASYMPTOMATIC]:
            self.ay_to_r()
        elif self.immunity_period is not None and self.status == State.REMOVED:
            self.r_to_s()
        self.days_passed += 1


if __name__ == "__main__":
    model_parameters = {
        'population size': 10000,
        'initial outbreak size': 10,
        'alpha': 0.7,
        'spread_chance': 0.005,
        'EAY': 1/5,
        'AR': 1/5,
        'YR': 1/5,
        'immunity period': 100
    }

    def social_distancing_func(t):
        if t > 20 and t < 60:
            return 0.2
        else:
            return 1

    graph = nx.powerlaw_cluster_graph(model_parameters['population size'], 100, 0.01)
    logger.info("Initialized graph")
    #model = Population(graph, model_parameters)
    model = Population(graph, model_parameters, social_distancing_func)
    model.run(300)
    df = model.datacollector.get_model_vars_dataframe()
    df.plot()
    plt.title("Single population model")
    plt.grid(b=True, which='both')
    #plt.minorticks_on()
    plt.show()
