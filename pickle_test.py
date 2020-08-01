import pickle
import matplotlib.pyplot as plt
import networkx as nx
import SEIR
import pandas as pd

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)

pd.options.plotting.backend = "plotly"

model_parameters = {
    'population size': 10000,
    'initial outbreak size': 10,
    'alpha': 0.7,
    'spread_chance': 0.005,
    'EAY': 1 / 5,
    'AR': 1 / 5,
    'YR': 1 / 5,
    'death rate': 0.01,
    'immunity period': None
}

model_steps = 10
graph = nx.powerlaw_cluster_graph(model_parameters['population size'], 100, 0.01, seed=5)
logger.info("Initialized graph")
model = SEIR.Population(graph, model_parameters)
model.run(model_steps)

#model.clear_social_distancing_func()
#pickle.dump(model, open( "/home/vbasu/Documents/SEYAR_models_sv/test.p", "wb"))

model.save_model("/home/vbasu/Documents/SEYAR_models_sv/testing")