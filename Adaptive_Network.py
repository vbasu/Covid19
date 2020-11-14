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


class AdaptiveModel:
    """
    An agent based model using the SEYAR flow and an adaptive network
    """

    def __init__(self, num_agents: int):
        self.num_agents = num_agents


class Agent:
    """
    An agent of the adaptive model
    """
    def __init__(self, status):
        pass