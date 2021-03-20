
import pymc3 as pm




iterations = 3000
learning_rate = 0.01
clique_size = [5, 5]
no_of_filters = 24
mini_batch_size = 200 
hybrid_monte_carlo_leap_size = 30
no_hops_mcmc = 1
# We want to minimize the KL divergence between
# Distribution of samples from the dataset, and distribution represented by our modeld

def load_data(path):
    return None
def getMiniBatchData(data, mini_batch_size):
    batches = sklearn.model_selection.ShuffleSplit(data, mini_batch_size)
    return batches
def HybridMonteCarlo(mini_batch_data):
    states = []
    for i in range(no_hops_mcmc):
        current = pm.sample(1, step=pm.hmc(), random_seed=123, progressbar=True)
        states.append(current)
    return states
def derive(data):
    return None

data = load_data("../../")
weights = np.zeros()
model = np.zeros()
for iteration in range(iterations):

    # Get minibatch of 200 images from dataset
    mini_batch_data = getMiniBatchData(data, mini_batch_size)

    # Gibbs sampling
    sampled_data = HybridMonteCarlo(mini_batch_data)

    # Derivative wrt equilibrium dist over visible variables (our model)
    deriv_wrt_model = derive(model)

    # Derivative wrt sampled data 
    deriv_wrt_sampled_data = derive(sampled_data)

    # Update weights according to KL divergence
    weights = weights + learning_rate * (deriv_wrt_sampled_data - deriv_wrt_model)

 