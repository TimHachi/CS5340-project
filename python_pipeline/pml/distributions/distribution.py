import math

class distribution():
    '''
    EVAL - Evaluate, i.e. compute, the probability
    '''
    def eval(self, x):
        pass
    '''
    SAMPLE - Draw i.i.d. samples from the probability
    '''
    def sample(self, nsamples):
        pass
    '''
    UNNORM - Evaluate, i.e. compute, the unnormalized probability
    '''
    def unnorm(self, x):
        return self.eval(x)

    '''
    LOG - Evaluate, i.e. compute, the log probability
    '''
    def log(self, x):
        return math.log(self.eval(x))

    '''
    ENERGY - Evaluate, i.e. compute, the energy
    '''
    def energy(self, x):
        return -math.log(self.unnorm(x))


