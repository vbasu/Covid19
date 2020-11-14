import SEIR
import networkx as nx
import matplotlib.pyplot as plt

import dill
import pickle

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_model(filename):
    """
    Given a filename (no extension), reload the metapopulation model
    :param filename:
    :return: metapopualtion model
    """
    with open(filename + ".dil", 'rb') as f:
        m = dill.load(f)
    '''
    with open(filename + ".pkl", 'rb') as f:
        model = pickle.load(f)
    for population in model.populations:
        population.reinstate_social_distancing_func(sdf)
    '''
    return m


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

    def save_model(self, filename):
        """
        Saves a metapopulation model to a pkl and dil file
        :param filename: name of the file (without extension) to save to
        :return:  none
        """
        with open(filename + ".dil", 'wb') as f:
            dill.dump(self, f)
            #dill.dump(self.populations[0].social_distancing_func, f)

        '''
        for population in populations:
            population.clear_social_distancing_func()
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)
        '''

if __name__ == "__main__":
    cross_influence_matrix = [[0, 0.1],
                              [0.1, 0]]

    populations = []
    for i in range(2):
        model_parameters = {
            'population size': 10000,
            'initial outbreak size': 10,
            'alpha': 0.7,
            'spread_chance': 0.005,
            'EAY': 1 / 5,
            'AR': 1 / 5,
            'YR': 1 / 5,
            'death rate': 0,
            'immunity period': 60
        }
        if i > 0:  # Let's make the second one have no initial outbreak
            model_parameters['spread_chance'] = 0.0005
        graph = nx.powerlaw_cluster_graph(model_parameters['population size'], 100, 0.01)
        logger.info("Initialized graph")
        model = SEIR.Population(graph, model_parameters)

        populations.append(model)

    meta_population = MetaPopulation(populations, cross_influence_matrix)
    meta_population.run(100)

    meta_population.save_model("Models/meta_population_test")
    '''
    fig, axs = plt.subplots(len(cross_influence_matrix))
    dfs = [pop.datacollector.get_model_vars_dataframe() for pop in meta_population.populations]
    for ax, df in zip(axs,dfs):
        df.plot(ax=ax, grid=True)
        ax.set_title("Subpopulation")
        #ax.set_yscale('log')

    #plt.yscale('log')
    #fig.suptitle("Metapopulation Model")
    fig.tight_layout(pad=1.0)
    #plt.grid(b=True, which='major')
    plt.show()
    '''
