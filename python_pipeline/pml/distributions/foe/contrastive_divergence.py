

iterations = 10000;

# We want to minimize the KL divergence between
# Distribution of samples from the dataset, and distribution represented by our model


for iteration in range(iterations):



    # Gibbs sampling



    # Derivative wrt equilibrium dist over visible variables (our model)
    pos =   

    # Derivative wrt sampled data 
    neg = 

    weights = weights + learning_rate * (pos + neg)