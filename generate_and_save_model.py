import networkx as nx
import SEIR
import pandas as pd

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
    'death rate': 0.01,
    'immunity period': None,
}


def get_metric(df):
    return df['Removed'].max()

def social_distancing_func(t):
    if 15 < t:
        return 0.4
    else:
        return 1


graph = nx.powerlaw_cluster_graph(model_parameters['population size'], 100, 0.01, seed=5)
logger.info("Initialized graph")
max_val = -1
min_val = 10001
for i in range(30):
    print( "starting model" + str(i) + "\n")
    model = SEIR.Population(graph, model_parameters)
    model.run_until_stable()
    df = model.get_data()
    metric = get_metric(df)
    if metric < min_val:
        model.save_model("Models/variation_test_min")
        min_val = metric
    if metric > max_val:
        model.save_model("Models/variation_test_max")
        max_val = metric

#filename = "variation_seed_2"
#model.save_model("Models/" + filename)


