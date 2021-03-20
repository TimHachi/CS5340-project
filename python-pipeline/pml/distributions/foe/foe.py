class foe:

    # Abstract constants
    default_expert

    # Internal parameters used by the model
    epsilon = 1.e-8
    experts = []
    imdims = [3,3]
    conv_method = 'valid'
    update_filter_matrices = True
    zeromean_filter = True
    conditional_sampling = False

    # Internal properties   
    A_ = eye(9);
    filter_size_ = [3, 3]
    J_tilde_ = [1 -1 zeros(1,7); 1 0 0 -1 zeros(1,5)]';
    J = [1 -1 zeros(1,7); 1 0 0 -1 zeros(1,5)]
    filter_matrices_uptodate = False
    conv2 = []
    imfilter = []
    makemat = []
    filter_matrices = []
    
    # Convenience properties   
    nexperts = 0
    nfilters = 0
    J_tilde = 0
    A = 0

    def __init__(self):
    def set_imdims():
    def set_experts():
    def get_nexperts():
        return len(self.experts)
    def get_nfilters():
        return max(2, len(self.filter))
    def img_cliques():
        
    def set_filter():
    def get_A():
    def get_J_tilde():
    def filter_size():
    def update():
    def create_filter_matrices():
        filter_matrices = np.zeros([1, self.nfilters])
        for i in range(self.nfilters):
            filter_matrices[i] = make_mat(filter)
 