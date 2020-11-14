import SEIR
import networkx as nx
import matplotlib.pyplot as plt
from MetaPopulation import *

import dill
import pickle

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)
cross_influence_matrix = [[0, 0],
                          [0, 0]]


def social_distancing_func(t):
    if t < 60:
        return 0.1
    else:
        return 1


populations = []
model_parameters = {
    'population size': 10000,
    'initial outbreak size': 100,
    'alpha': 0.7,
    'spread_chance': 0.005,
    'EAY': 1 / 5,
    'AR': 1 / 5,
    'YR': 1 / 5,
    'death rate': 0,
    'immunity period': 20
}


# Create the first population
graph = nx.powerlaw_cluster_graph(model_parameters['population size'], 100, 0.01)
logger.info("Initialized graph")
model = SEIR.Population(graph, model_parameters)
populations.append(model)

# Make adjustments on the second population
graph_2 = nx.powerlaw_cluster_graph(model_parameters['population size'], 100, 0.01)
model_2 = SEIR.Population(graph_2, model_parameters, social_distancing_func)
populations.append(model_2)

meta_population = MetaPopulation(populations, cross_influence_matrix)
meta_population.run(120)

meta_population.save_model("Models/meta_population_osc_control_lowimun")
