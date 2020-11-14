# A class to implement the SEYAR agent based model avoiding the costs of Mesa

from enum import IntEnum
import networkx as nx
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import time

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)


class State(IntEnum):
    SUSCEPTIBLE = 0
    EXPOSED = 1
    ASYMPTOMATIC = 2
    SYMPTOMATIC = 3
    RECOVERED = 4


class SEYARModel:
    """
    An agent based model to simulate the spread of Covid-19
    """

    def __init__(self,
                 num_agents: int,
                 initial_outbreak_size: int,
                 graph: nx.Graph,
                 spread_chance: float,
                 p_eay: float,
                 alpha: float,
                 p_ar: float,
                 p_yr: float):
        """
        Constructor for the SEYAR model

        :param num_agents: number of agents in the model
        :param initial_outbreak_size: number of agents initially infected
        :param graph: the networkx graph that defines interaction between the agents
        :param spread_chance: the probability of transmission given contact
        :param p_eay: the rate at which agents progress from Exposed to Infected
        :param alpha: the proportion of infected agents which are asymptomatic
        :param p_ar: the rate at which agents progress from Asymptomatic to Recovered
        :param p_yr: the rate at which agents progress from Symptomatic to Recovered
        """
        self.num_agents = num_agents
        self.initial_outbreak_size = initial_outbreak_size
        self.graph = graph
        self.spread_chance = spread_chance
        self.p_eay = p_eay
        self.alpha = alpha
        self.p_ar = p_ar
        self.p_yr = p_yr

        self.states = np.zeros((num_agents,)) + State.SUSCEPTIBLE  # The state of each agent in the model

        # Randomly pick the initial outbreak set
        initial_infected = np.random.choice(num_agents, initial_outbreak_size, replace=False)
        for agent in initial_infected:
            self.states[agent] = State.EXPOSED

        self.history = [self.states]  # The stored history of the states

        logger.info("Getting the adjacency matrix...\n")
        #self.adjacency_matrix = nx.adjacency_matrix(self.graph)
        self.adjacency_matrix = nx.to_numpy_array(self.graph)
        logger.info("Got the adjacency matrix.\n")

    def step(self):
        """
        Run one step in the model.
        :return: None
        """
        s_to_e = self.find_newly_infected()
        e_to_a, e_to_y = self.get_e_to_ay_transitions()
        a_to_r = self.get_ay_to_r_transitions(State.ASYMPTOMATIC)
        y_to_r = self.get_ay_to_r_transitions(State.SYMPTOMATIC)
        non_transitioning_agents = np.logical_not(np.logical_or.reduce((s_to_e, e_to_a, a_to_r, y_to_r)))

        new_states = non_transitioning_agents * self.states \
            + s_to_e * State.EXPOSED \
            + e_to_a * State.ASYMPTOMATIC \
            + e_to_y * State.SYMPTOMATIC \
            + a_to_r * State.RECOVERED \
            + y_to_r * State.RECOVERED

        self.states = new_states

        # Add the current set of states to the history
        self.history.append(self.states)

    def find_newly_infected(self) -> np.ndarray:
        """
        Finds the agents which should advance from S to E. Does not apply the change.
        :return: vector to_be_infected where to_be_infected[i] = 1 iff agent i will transition from S to E
        """
        # Find the contagious agents
        infected_agents = np.logical_or(self.states == State.ASYMPTOMATIC, self.states == State.SYMPTOMATIC)
        # For each agent count how many neighbors are contagious
        infected_neighbor_counts = np.dot(self.adjacency_matrix, infected_agents)
        infection_probabilities = self.get_infection_probabilities(infected_neighbor_counts)
        to_be_infected = infection_probabilities > np.random.rand(self.num_agents)
        # Filter out non-susceptible agents
        to_be_infected = to_be_infected * (self.states == State.SUSCEPTIBLE)
        return to_be_infected

    def get_e_to_ay_transitions(self) -> (np.ndarray, np.ndarray):
        """
        Finds which exposed agents should move to symptomatic and asymptomatic
        :return: None
        """
        exposed_agents = self.states == State.EXPOSED
        advancing_agents = exposed_agents * self.p_eay > np.random.rand(self.num_agents)
        split = np.random.rand(self.num_agents)
        e_to_a = advancing_agents * (split <= self.alpha)
        e_to_y = advancing_agents * (split > self.alpha)
        return e_to_a, e_to_y

    def get_ay_to_r_transitions(self, state) -> np.ndarray:
        """
        Finds which infected agents should recover for either asymptomatic or symptomatic
        :return:
        """
        infected_agents = self.states == state
        advancing_agents = infected_agents * self.p_ar > np.random.rand(self.num_agents)
        return advancing_agents

    def get_infection_probabilities(self, counts) -> np.ndarray:
        """
        Given a vector counting contagious neighbors for each agent, return
        :param counts: counts[i] is the number of contagious agents adjacent to agent i
        :return: a vector
        """
        #np_array = counts.toarray()
        return 1 - np.power(1 - self.spread_chance, counts)
        #return 1 - sps.csr_matrix.power(counts, 1 - self.spread_chance)

    def run(self, num_steps) -> None:
        for i in range(num_steps):
            logger.info("Steps completed:" + str(i))
            self.step()

    def test_plot(self) -> None:
        data = [np.sum(day_state == State.SUSCEPTIBLE) for day_state in self.history]
        plt.plot(data)
        plt.show()


if __name__ == '__main__':
    num_agents = 10000
    graph = nx.powerlaw_cluster_graph(num_agents, 100, 0.01)
    #graph = nx.barabasi_albert_graph(n=num_agents, m=9)
    print("Graph initialized...\n")
    model = SEYARModel(num_agents, 10, graph, 0.01, 0.2, 0.7, 0.2, 0.2)
    tic = time.time()
    model.run(50)
    toc = time.time()
    print("Took this many seconds to run: " + str(toc-tic))
    model.test_plot()