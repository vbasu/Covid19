import time

import networkx as nx
import SEIR
import pandas as pd
import matplotlib.pyplot as plt

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)

model_parameters = {
    'population size': 10000,
    'initial outbreak size': 10,
    'alpha': 0.7,
    'spread_chance': 0.005,
    'EAY': 1 / 5,
    'AR': 1 / 5,
    'YR': 1 / 5,
    'death rate': 0,
    'immunity period': None,
}


def get_metric(df):
    return df['Recovered'].max()

graph = nx.powerlaw_cluster_graph(model_parameters['population size'], 100, 0.01, seed=5)
tic = time.time()
model = SEIR.Population(graph, model_parameters)
model.run(25)
df = model.get_data()
metric = get_metric(df)
toc = time.time()
print("Time taken for model: %f\n" % (toc-tic))
print("Total agents infected: %d\n" % metric)

df.plot()
plt.show()
#filename = "variation_seed_2"
#model.save_model("Models/" + filename)


