import matplotlib.pyplot as plt
import networkx as nx
import SEIR
import MetaPopulation
import pandas as pd

#pd.options.plotting.backend = "plotly"

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)
cross_influence_matrix = [[0, 0.1, 0.2],
                          [0.1, 0, 0.3],
                          [0.2, 0.3, 0]]

populations = []
model_parameters_0 = {
    'population size': 10000,
    'initial outbreak size': 10,
    'alpha': 0.7,
    'spread_chance': 0.001,
    'EAY': 1 / 10,
    'AR': 1 / 10,
    'YR': 1 / 10,
    'death rate': 0,
    'immunity period': None
}

model_parameters_1 = {
    'population size': 10000,
    'initial outbreak size': 10,
    'alpha': 0.7,
    'spread_chance': 0.001,
    'EAY': 1 / 10,
    'AR': 1 / 10,
    'YR': 1 / 10,
    'death rate': 0,
    'immunity period': None
}

model_parameters_2 = {
    'population size': 10000,
    'initial outbreak size': 0,
    'alpha': 0.7,
    'spread_chance': 0.001,
    'EAY': 1 / 10,
    'AR': 1 / 10,
    'YR': 1 / 10,
    'death rate': 0,
    'immunity period': None
}

model_parameters = [model_parameters_0, model_parameters_1, model_parameters_2]
for mp in model_parameters:
    graph = nx.powerlaw_cluster_graph(mp['population size'], 100, 0.01)
    logger.info("Initialized graph")
    model = SEIR.Population(graph, mp)

    populations.append(model)

meta_population = MetaPopulation.MetaPopulation(populations, cross_influence_matrix)
meta_population.run(100)

fig, axs = plt.subplots(len(cross_influence_matrix))
dfs = [pop.datacollector.get_model_vars_dataframe() for pop in meta_population.populations]
for ax, df in zip(axs, dfs):
    df.plot(ax=ax, grid=True)
    ax.set_title("Subpopulation")
    # ax.set_yscale('log')

# plt.yscale('log')
# fig.suptitle("Metapopulation Model")
fig.tight_layout(pad=1.0)
# plt.grid(b=True, which='major')
plt.show()
