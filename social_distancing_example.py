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


def social_distancing_func_simple(t):
    if 15 < t < 25:
        return 0.4
    else:
        return 1


model_steps = 50
graph = nx.powerlaw_cluster_graph(model_parameters['population size'], 100, 0.01, seed=5)
logger.info("Initialized graph")
model = SEIR.Population(graph, model_parameters, social_distancing_func_simple)
model.run(model_steps)
df = model.datacollector.get_model_vars_dataframe()
fig = df.plot()
#fig.show()
model2 = SEIR.Population(graph, model_parameters)
model2.run(model_steps)
df2 = model2.datacollector.get_model_vars_dataframe()
#fig, axs = plt.subplots(2)
ax = df.plot()
#df2.plot(ax=ax, grid=True, linestyle='--')
ax2 = df2.plot()
ax.show()
ax2.show()
#sdf_values = [social_distancing_func_simple(t) for t in range(0, model_steps)]
#axs[1].plot(sdf_values)
#axs[1].set_ylim(0, 1)
#axs[1].grid(True)
title = 'alpha:{alpha}, spread chance:{spread_chance},\n EAY:{EAY}, AR:{AR}, YR:{YR}, immunity:{immunity period}'.format(
    **model_parameters)
plt.title(title)
# plt.grid(b=True, which='both')
# plt.axvline(x=30, color='k', linestyle='--')
# plt.yscale('log')
# plt.minorticks_on()
# plt.show()
