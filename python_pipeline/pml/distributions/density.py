from .distribution import *

class density(distribution):
    '''
    EVAL - Evaluate, i.e. compute, the probability
    '''
    def eval(self, x):
        pass
    '''
    ENERGY - Evaluate, i.e. compute, the energy
    '''
    def energy(self, x):
        return -math.log(self.unnorm(x))
    '''
    LOG_GRAD_X - Compute the gradient of the log density 
    '''
    def log_grad_x(self, x):
        print("Function not implemented!")
    '''
    SAMPLE - Draw i.i.d. samples from the probability density
    '''
    def sample(self, nsamples):
        print("Function not implemented!")
