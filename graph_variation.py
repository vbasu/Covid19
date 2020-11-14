import networkx as nx
import SEIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)

# Hold this constant throughout
model_parameters = {
    'population size': 10000,
    'initial outbreak size': 10,
    'alpha': 0.7,
    'spread_chance': 0.005,
    'EAY': 1 / 5,
    'AR': 1 / 5,
    'YR': 1 / 5,
    'death rate': 0.01,
    'immunity period': None,
}


def get_metric(df):
    return df['Removed'].max()


p_values = np.linspace(0, 1, num=2)
removed_count = []
for p in p_values:
# m_values = [5*x for x in range(1, 50)]
# for m in m_values:
    logger.info("Starting simulation for m = " + str(p) + "\n")
    graph = nx.powerlaw_cluster_graph(model_parameters['population size'], 100, p, seed=5)
    logger.info("Initialized graph")
    model = SEIR.Population(graph, model_parameters)
    model.run(50)
    df = model.get_data()
    removed_count.append(get_metric(df))

print(removed_count)
plt.plot(p_values, removed_count)
plt.show()




