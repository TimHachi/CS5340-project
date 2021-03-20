iterations = 3000
learning_rate = 0.01
clique_size = [5, 5]
no_of_filters = 24
mini_batch_size = 200 
# We want to minimize the KL divergence between
# Distribution of samples from the dataset, and distribution represented by our modeld

data = load_data("../../")

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
    weights = weights + learning_rate * (pos + neg)

