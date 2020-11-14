from seirsplus.models import *
from seirsplus.networks import *
import networkx

numNodes = 100000
baseGraph = networkx.barabasi_albert_graph(n=numNodes, m=9)
G_normal = custom_exponential_graph(baseGraph, scale=100)
G_distancing = custom_exponential_graph(baseGraph, scale=10)
G_quarantine = custom_exponential_graph(baseGraph, scale=5)

model = SEIRSNetworkModel(G=G_normal, beta=0.155, sigma=1/5.2, gamma=1/12.39, mu_I=0.0004, p=0.5,
                          G_Q=G_quarantine, beta_Q=0.155, sigma_Q=1/5.2, gamma_Q=1/12.39, mu_Q=0.0004,
                          theta_E=0.02, theta_I=0.02, phi_E=0.2, phi_I=0.2, psi_E=1.0, psi_I=1.0, q=0.5, initI=10)

checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5], 'theta_E': [0.02, 0.02], 'theta_I': [0.02, 0.02], 'phi_E':   [0.2, 0.2], 'phi_I':   [0.2, 0.2]}

model.run(T=300, checkpoints=checkpoints)
model.figure_infections()