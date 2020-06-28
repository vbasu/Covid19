import SEIR
import networkx as nx
import matplotlib.pyplot as plt

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MetaPopulation():
    """
    Contains a group of populations with some cross-influence

    Diagonal entries on the cross influence matrix (self-influence) should be zero
    """
    def __init__(self, populations, cross_influence_matrix):
        self.populations = populations
        self.cross_influence_matrix = cross_influence_matrix

    def step(self):
        ratios = [SEIR.count_exposed(population) / population.population_size for population in self.populations]
        for population in self.populations:
            population.step()
        self.cross_influence(ratios)

    def cross_influence(self, ratios):
        """
        Do the cross influence
        :param ratios: ratios of exposed agents in other populations
        :return: None
        """
        for i, population in enumerate(self.populations):
            population.cross_influence(self.cross_influence_matrix[i], ratios)

    def run(self, steps):
        """
        Runs the model
        :param steps: number of steps to run for
        :return: None
        """
        for i in range(steps):
            logger.info("Steps Completed: " + str(i))
            self.step()

if __name__ == "__main__":

    cross_influence_matrix = [[0, 0.5],
                              [0.5, 0]]

    populations = []
    for _ in range(2):
        model_parameters = {
            'population_size': 10000,
            'initial_outbreak_size': 10,
            'alpha': 0.7,
            'spread_chance': 0.005,
            'EAY': 1/5,
            'AR': 1/5,
            'YR': 1/5,
        }

        graph = nx.powerlaw_cluster_graph(model_parameters['population_size'], 100, 0.01)
        logger.info("Initialized graph")
        model = SEIR.Population(graph, model_parameters)

        populations.append(model)

    meta_population = MetaPopulation(populations, cross_influence_matrix)
    meta_population.run(60)

    fig, axs = plt.subplots(len(cross_influence_matrix))
    dfs = [pop.datacollector.get_model_vars_dataframe() for pop in meta_population.populations]
    for ax, df in zip(axs,dfs):
        df.plot(ax=ax, grid=True)
        ax.set_title("Subpopulation")

    fig.suptitle("Metapopulation Model")
    fig.tight_layout(pad=1.0)
    #plt.grid(b=True, which='major')
    plt.show()
