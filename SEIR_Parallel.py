from enum import Enum

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import pickle
import dill

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelActivation(SimultaneousActivation):
    """
    A scheduler to parallelize the model stepping?
    """
    def step(self) -> None:
        pool = mp.Pool(mp.cpu_count())
        agent_keys = list(self._agents.keys())
        for agent_key in agent_keys:
            pool.apply_async(self._agents[agent_key].step(), )
        pool.close()
        pool.join()

class State(Enum):
    SUSCEPTIBLE = 0
    EXPOSED = 1
    ASYMPTOMATIC = 2
    SYMPTOMATIC = 3
    RECOVERED = 4
    REMOVED = 5


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


def count_recovered(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.RECOVERED])


def count_diseased(model):
    return count_exposed(model) + count_asymptomatic(model) + count_symptomatic(model)


def count_removed(model):
    statuses = [agent.status for agent in model.schedule.agents]
    return len([x for x in statuses if x == State.REMOVED])


def load_model(filename):
    """
    Loads a model
    :param filename: file path WITHOUT THE EXTENSION
    :return: the Population object for the model
    """
    with open(filename + ".dil", 'rb') as f:
        sdf = dill.load(f)
    with open(filename + ".pkl", 'rb') as f:
        model = pickle.load(f)
    model.reinstate_social_distancing_func(sdf)
    return model


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

        self.datacollector = DataCollector({  # "Exposed": count_exposed,
            "Susceptible": count_susceptible,
            "Recovered": count_recovered,
            # "Asymptomatic": count_asymptomatic,
            # "Symptomatic": count_symptomatic,
            "Diseased": count_diseased,
            "Removed": count_removed
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

    def run_until_stable(self):
        max_steps = 1e3
        steps = 0
        window_size = 50
        while steps < max_steps:
            self.step()
            logger.info("Steps Completed:" + str(steps))
            if steps > window_size:
                data = self.get_data()
                last_value = int(data.tail(1)['Diseased'])
                if last_value == 0:
                    break
                window_average = np.mean(data.tail(window_size)['Diseased'])  # Window for determining stopping rule
                if abs(last_value - window_average) / window_average < 0.005:
                    break
            steps += 1

    def cross_influence(self, influence_coefficients, ratios):
        for inf_coeff, ratio in zip(influence_coefficients, ratios):
            susceptibles = list(filter(lambda x: x.status == State.SUSCEPTIBLE, self.grid.get_all_cell_contents()))
            to_flip = self.random.sample(susceptibles, int(inf_coeff*ratio*len(susceptibles)))
            for agent in to_flip:
                agent.status = State.EXPOSED

    def clear_social_distancing_func(self):
        """
        Clears the social distancing function of the model and all its agents for pickling
        :return: None
        """
        self.social_distancing_func = None
        for agent in self.grid.get_all_cell_contents():
            agent.social_distancing_func = None

    def reinstate_social_distancing_func(self, social_distancing_func=lambda x: 1):
        """
        Re-adds the social distancing func to the model and all its agents
        :param social_distancing_func: social distancing func to be re-added
        :return: None
        """
        self.social_distancing_func = social_distancing_func
        for agent in self.grid.get_all_cell_contents():
            agent.social_distancing_func = social_distancing_func

    def save_model(self, filename):
        """
        Save the model to a pickle and dill file
        :param filename: filename (without extension) to save to
        :return: None
        """
        with open(filename + ".dil", 'wb') as f:
            dill.dump(self.social_distancing_func, f)
        self.clear_social_distancing_func()
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    def get_data(self):
        return self.datacollector.get_model_vars_dataframe()

    '''
    def plot_social_distancing_func(self):
        sdf_values = [social_distancing_func_simple(t) for t in range(0, len(self.get_data()))]
        fig, ax = plt.subplots()
        ax.plot(range(0, self.get_data().count), sdf_values)
        plt.show()
    '''


class Person(Agent):
    def __init__(self, unique_id, model, status, parameters, social_distancing_func):
        super().__init__(unique_id, model)
        # self.random.seed(2)
        # np.random.seed(2)
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
        self.death_rate = parameters['death rate']

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
                self.status = State.RECOVERED
        elif self.status == State.SYMPTOMATIC:
            if self.random.random() < self.p_YR:
                self.status = State.RECOVERED
        self.remove_agent()

    def r_to_s(self):
        if self.random.random() < 1/self.immunity_period:
            self.status = State.SUSCEPTIBLE

    def remove_agent(self):
        if self.random.random() < self.death_rate:
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
        elif self.immunity_period is not None and self.status == State.RECOVERED:
            self.r_to_s()
        self.days_passed += 1


