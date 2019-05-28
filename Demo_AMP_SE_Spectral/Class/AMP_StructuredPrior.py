from Library.import_library import *
from Functions.functions_save_load import save_object, load_object
from Functions.generic_functions import sign, relu
from Class.fout_fu_relu import fout_gout_dgout_relu, fu_0_1_2_relu 
from Class.fout_fu_sign import fout_gout_dgout_sign, fu_0_1_2_sign

class AMP(object):
    def __init__(self, K=1, N=1000, alpha_1=1, alpha_2=1, non_linearity_1='linear', non_linearity_2='linear', weights='gaussian', Delta_1=0.01, Delta_2=0, method_gout='explicit', seed=False, initialization_mode='planted', model='VU', verbose ='True'):
        ## Scaling
        self.scaling = 'stpr' # 'stpr' or 'nn'  # 1/sqrt(M) or 1/sqrt(N) in first layer
        ## Model
        self.model = model  # 'UU' or 'VU'
        ## Parameters
        self.K, self.N = K, N
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2  # alpha_1 = P/M (stpr) or P/N (nn) , alpha_2 = M/N
        ## Limit alpha_2 = 0 : no prior
        self.with_structured_prior = True
        if self.alpha_2 == 0:
            self.with_structured_prior = False
            self.alpha_2 = 1 
        ## Change definition of P depending on the scaling
        self.P = int(self.N * self.alpha_1 * self.alpha_2)
        self.M = int(self.N * self.alpha_2)
        ## Non linearities and noises
        self.non_linearity_1 = non_linearity_1
        self.non_linearity_2 = non_linearity_2
        self.Delta_1, self.Delta_2 = Delta_1, Delta_2
        self.Delta_2_mat = self.Delta_2 * np.identity(self.K)
        ## Seed
        self.seed = seed
        np.random.seed(0) if self.seed else 0
        ## verbose 
        self.verbose = verbose

        ## Test set parameters
        self.number_batch_test, self.batch_size_test = 10, 10000
        self.M_test = self.number_batch_test * self.batch_size_test  # size test_set
        ## Data: gaussian with zero mean and variance 1
        self.lambda_X, self.Sigma_X = 0, 1
        ## Weights
        self.weights = weights  # 'binary' or 'gaussian'
        self.lambda_v, self.Sigma_v = np.zeros((self.K)), np.identity(
            self.K)  # mean and covariance for v
        self.lambda_u, self.Sigma_u = np.zeros((self.K)), np.identity(
            self.K)  # mean and covariance for v
        self.lambda_w, self.Sigma_w = np.zeros((self.K)), np.identity(
            self.K)  # mean and covariance for w

        # Option for computing g_out
        self.method_gout = method_gout
        if self.method_gout == 'explicit':
            self.MC_activated = False
            self.print_gout = False
        elif self.method_gout == 'MC':
            self.MC_activated = True
            self.print_gout = True
        else:
            raise NameError('Method to compute g_out undefined')
        #print(f'Method for gout: {self.method_gout}') if self.verbose else 0
        ## MC: Number of iterations Monte Carlo, Number of procs Monte Carlo
        self.N_iter_MC, self.NProcs_MC = 10000, 2         
        ## Average overlaps
        self.list_average_m_q = []
        self.average_m_q, self.std_m_q = 0, 0
        self.N_average = 0
        self.gen_error, self.gen_error_push_forward = 0, 0
        ## Convergence of the algorithm
        self.min_step_AMP, self.max_step_AMP = 10, 100
        self.diff_mode = 'overlap'  # 'overlap' or 'weight'
        self.threshold_error_overlap, self.threshold_error_weight, self.threshold_bayes = 1e-4, 1e-4, 5e-2
        ## Initialization near the ground truth solution
        self.initialization_mode = initialization_mode  # random , planted, average
        #print(
        #    f'Initialization: {self.initialization_mode}') if self.verbose else 0
        self.planted_noise, self.coef_initialization_random = 1, 1
        ## Nishimori_identity and Average dgout
        self.Nishimori_identity, self.averaged_dgout = False, False
        ## Damping
        self.damping_activated, self.damping_coef = True, 0.05
        #print(f'Damping: {self.damping_activated} / {self.damping_coef}')


        ## Plot Options
        self.plot_Fig, self.save_Fig = False, False
        ## Option prints
        self.print_Initialization, self.print_Running, self.print_Errors = False, False, False
        ## Directory
        self.directory = self.non_linearity_1 + '_' + self.non_linearity_2 + '_' + 'K=' + str(self.K) + '/'
        self.data_directory = 'Data/AMP_SeqToSeq' + '_' + self.model +'/' + self.directory
        ## Create paths
        os.makedirs(self.data_directory) if not os.path.exists(self.data_directory) else 0
        ## Filename
        self.name = 'AMP_SeqToSeq_'+self.model+'_K='+str(self.K)+'_a1=' + '%.4f' % self.alpha_1+'_a2='+'%.4f' % self.alpha_2 + '_d1=' + '%.6f' % self.Delta_1 + \
            '_d2='+'%.6f' % self.Delta_2 + '_'+self.non_linearity_1 + '_' + \
            self.non_linearity_2 + '_' + self.weights + '_N=' + \
            str(self.N) + '_' + self.initialization_mode + '_' + self.scaling
        self.file_name = self.data_directory + self.name + '.pkl'

        self.generate_parameters()
        self.change_size_overflow()

    def change_size_overflow(self):
        """
        Change size if alpha too large to avoid memory overflow or unprecisison
        """
        self.PM_max = 1e8
        if (self.M * self.P > self.PM_max):
            if self.scaling == 'stpr':
                if self.alpha_1 * self.alpha_2 > 1:
                    self.N = int(
                        sqrt(self.PM_max / (self.alpha_1 * self.alpha_2**2)))
                else:
                    self.N = int(sqrt(self.PM_max / (self.alpha_2)))
                self.P, self.M = int(
                    self.N * self.alpha_1 * self.alpha_2), int(self.N * self.alpha_2)
            print(
                f"Change size to avoid memory overflow: N={self.N}, P={self.P}, M={self.M}")

    def generate_parameters(self):
        """
        Generate dict of usfeull parameters
        """
        self.dict_parameters = {}
        self.dict_parameters['K'] = self.K
        self.dict_parameters['N'] = self.N
        self.dict_parameters['M'] = self.M
        self.dict_parameters['P'] = self.P
        self.dict_parameters['alpha_1'] = self.alpha_1
        self.dict_parameters['alpha_2'] = self.alpha_2
        self.dict_parameters['Delta_1'] = self.Delta_1
        self.dict_parameters['Delta_2'] = self.Delta_2
        self.dict_parameters['non_linearity_1'] = self.non_linearity_1
        self.dict_parameters['non_linearity_2'] = self.non_linearity_2
        self.dict_parameters['model'] = self.model
        self.dict_parameters['weights'] = self.weights
        self.dict_parameters['method_gout'] = self.method_gout
        self.dict_parameters['seed'] = self.seed
        self.dict_parameters['N_average'] = self.N_average
        self.dict_parameters['damping_activated'] = self.damping_activated
        self.dict_parameters['damping_coef'] = self.damping_coef
        self.dict_parameters['MC_activated'] = self.MC_activated
        self.dict_parameters['Nishimori_identity'] = self.Nishimori_identity
        self.dict_parameters['averaged_dgout'] = self.averaged_dgout
        self.dict_parameters['threshold_error_overlap'] = self.threshold_error_overlap
        self.dict_parameters['threshold_error_weight'] = self.threshold_error_weight
        self.dict_parameters['diff_mode'] = self.diff_mode
        self.dict_parameters['max_step_AMP'] = self.max_step_AMP
        self.dict_parameters['min_step_AMP'] = self.min_step_AMP
        self.dict_parameters['initialization_mode'] = self.initialization_mode
        self.dict_parameters['file_name'] = self.file_name
        #print(self.dict_parameters)
        print('\n') if self.verbose else 0

    ############## Generate data ##############
    def generate_X(self, M_samples):
        """ 
        Generate iid Gaussian matrix of size N x M_samples
        """
        X = np.random.normal(self.lambda_X, self.Sigma_X, (self.N, M_samples))
        if self.print_Initialization:
            print('X=', X, '\n')
        return X

    def generate_W2_W1(self):
        """
        Generate teacher weights W1^*, W2^* with prior depending on self.weights
        """
        if self.weights == 'gaussian':
            for j in range(self.P):
                self.W1[:, j] = np.random.multivariate_normal(self.lambda_v, self.Sigma_v)
            for i in range(self.N):
                self.W2[:, i] = np.random.multivariate_normal(self.lambda_w, self.Sigma_w)
        elif self.weights == 'binary':
            self.W1 = 2 * np.random.randint(2, size=(self.K, self.P)) - 1
            self.W2 = 2 * np.random.randint(2, size=(self.K, self.N)) - 1
        else:
            raise NameError('Weights Prior weights not defined')
        if self.print_Initialization:
            print('W2=', self.W2, '\n')
            print('W1=', self.W1, '\n')
        return (self.W2, self.W1)

    def Phi_out_1(self, z1):
        """
        Returns Phi(z1) + N(0,Delta_1)
        """
        if self.non_linearity_1 == 'linear':
            resul = z1
        elif self.non_linearity_1 == 'sign':
            resul = sign(z1)
        elif self.non_linearity_1 == 'relu':
            resul = relu(z1)
        else:
            raise NameError('Non_linearity not defined')
        size = resul.shape
        noise = np.random.normal(0, sqrt(self.Delta_1), size)
        if self.model == 'UU':
            resul += ( noise + noise.T) / sqrt(2)
        elif self.model == 'VU'  :
            resul += noise
        return resul

    def Phi_out_2(self, z2):
        """
        Returns Phi(z2) + N(0,Delta_2)
        """
        if self.non_linearity_2 == 'linear':
            resul = z2
        elif self.non_linearity_2 == 'sign':
            resul = sign(z2)
        elif self.non_linearity_2 == 'relu':
            resul = relu(z2)
        else:
            raise NameError('Non_linearity not defined')
        size = resul.shape
        resul += np.random.normal(0, sqrt(self.Delta_2), size)
        return resul

    def generate_Y(self, X):
        """
        Generate output Y
        - self.model == 'UU' : Y = UU'
        - self.model == 'VU' : Y = VU'

        - self.scaling == 'stpr' : Y-> Y/sqrt(M)
        - self.scaling == 'nn' : Y-> Y/sqrt(N)
        """
        N, M = X.shape
        Z = self.W2.dot(X) / sqrt(self.N)
        U = self.Phi_out_2(Z).reshape(self.K, M)

        if self.model == 'VU':
            if self.scaling == 'stpr':
                Z1 = (self.W1.transpose()).dot(U) / sqrt(self.M)
            else:
                Z1 = (self.W1.transpose()).dot(U) / sqrt(self.N)
        elif self.model == 'UU':
            if self.scaling == 'stpr':
                Z1 = (U.transpose()).dot(U) / sqrt(self.M)
            else:
                Z1 = (U.transpose()).dot(U) / sqrt(self.N)

        Y = self.Phi_out_1(Z1)
        if self.print_Initialization:
            print('Y=', Y, '\n')
            print('X=', X.shape, 'W2=', self.W2.shape, 'Z=', Z.shape, 'U=',
                  U.shape, 'W1=', self.W1.shape, 'Z1=', Z1.shape, 'Y=', Y.shape)
            print('N=', self.N, 'M=', M, 'P=', self.P, 'K=', self.K)
        return (Y, U)

    def generate_S_R(self):
        """
        Defines S and R matrices
        """
        if self.non_linearity_1 == 'linear':
            self.generate_S_R_linear()
        else:
            raise NameError('R and S undefined for this non linearity')
        return 0

    def generate_S_R_linear(self):
        """
        Defines S and R if self.non_linearity_1 == 'linear'
        Add a threshold to avoid overflow
        """
        Delta_1 = max(self.Delta_1, 5e-3)
        #print('Use threshold on S and R: Delta_1=', Delta_1)
        self.S = self.Y / Delta_1
        self.S_square = np.square(self.S)
        self.R = - 1 / Delta_1 * np.ones((self.P, self.M)) + self.S_square

    def generate_training_set(self):
        """
        - Generates data matrix X 
        - Generates teacher weights W_1, W_2 / V, W
        - Generates Y and U
        - Generates S and R
        """
        self.X = self.generate_X(self.M)
        self.X_square = np.square(self.X)
        self.generate_W2_W1()
        self.Y, self.U = self.generate_Y(self.X)
        self.generate_S_R()

    def generate_test_set(self, M_test):
        """
        - Creates a test set of size M_test
        - Generates new output 
        - returns X_test, Y_test
        """
        X_test = self.generate_X(M_test)
        Y_test, U = self.generate_Y(X_test)
        return X_test, Y_test

    # Damping
    def damping(self, X_new, X_self):
        """
        if damping activated returns X_new
        else returns (1-self.damping_coef) * (X_new) + self.damping_coef * X_self
        """
        if not self.damping_activated:
            return X_new
        else :
            return (1-self.damping_coef) * (X_new) + self.damping_coef * X_self

    ############## Initialization ##############
    def initialization_storage(self):
        """
        Initialization arrays
        """
        # Data and ground truth
        self.X = np.zeros((self.N, self.M))
        self.Y = np.zeros((self.P, self.M))
        self.S = np.zeros((self.P, self.M))
        self.R = np.zeros((self.P, self.M))
        self.W1 = np.zeros((self.K, self.P))
        self.W2 = np.zeros((self.K, self.N))

        # Matrix factorization layer
        # Mean and variance Messages: U_hat, C_hat_U
        self.U_hat = np.zeros((self.K, self.M))
        self.C_hat_U = np.zeros((self.K, self.K, self.M))
        # Mean and variance Updates: B_U, A_U
        self.B_U = np.zeros((self.K, self.M))
        self.A_U = np.zeros((self.K, self.K, self.M))
        # Mean and variance Messages: V_hat, C_hat_V
        self.V_hat = np.zeros((self.K, self.P))
        self.C_hat_V = np.zeros((self.K, self.K, self.P))
        # Mean and variance Updates: B_V, A_V
        self.B_V = np.zeros((self.K, self.P))
        self.A_V = np.zeros((self.K, self.K, self.P))

        # Fully connected layer
        # Mean and variance Messages: W_hat, C_hat_W
        self.W_hat = np.zeros((self.K, self.N))
        self.C_hat_W = np.zeros((self.K, self.K, self.N))
        # Mean and variance Updates: sigma, lambda
        self.sigma = np.zeros((self.K, self.K, self.N))
        self.sigma_inv = np.zeros((self.K, self.K, self.N))
        self.lambd = np.zeros((self.K, self.N))
        ## V and omega
        self.V = np.zeros((self.K, self.K, self.M))
        self.V_inv = np.zeros((self.K, self.K, self.M))
        self.omega = np.zeros((self.K, self.M))
        ## gout and dgout
        self.gout = np.zeros((self.K, self.M))
        self.dgout = np.zeros((self.K, self.K, self.M))
        # Store Onsager terms
        self.V_hat_onsager = np.zeros((self.K, self.P))
        self.U_hat_onsager = np.zeros((self.K, self.M))
        self.B_U_onsager = np.zeros((self.K, self.M))
        self.A_U_onsager = np.zeros((self.K, self.K, self.M))

        # Overlaps
        self.list_evolution_m_q = []
        self.m_v, self.q_v, self.m_u, self.q_u, self.m_w, self.q_w = 0, 0, 0, 0, 0, 0
        self.tab_m_q = []

    def initialization(self):
        """
        Initialization of the messages
        """
        ###### Initilization ######
        # Initialization Vhat, Chat_V, Uhat, Chat_U, What, Chat_W at t=0
        self.initialization_Uhat_Chat_U()
        self.initialization_Vhat_Chat_V()
        if self.with_structured_prior:
            self.initialization_What_Chat_W()
        print('Initialization') if self.verbose else 0
        self.compute_overlap()
        # Initilization A_U, V, A_V at t=0 and B_U, B_V, omega WITHOUT ONSAGER term
        # Compute gout, dgout without A_U, B_U
        self.initialization_no_Onsager_terms = True
        self.initialization_B_A_U()
        self.initialization_B_A_V()
        if self.with_structured_prior:
            self.initialization_V_omega()
            self.initialization_gout_dgout()
            self.initialization_sigma_lambda()
        self.initialization_no_Onsager_terms = False
        # Store Onsager
        self.U_hat_onsager = deepcopy(self.U_hat)
        self.V_hat_onsager = deepcopy(self.V_hat)
        self.W_hat_onsager = deepcopy(self.W_hat)
        self.B_U_onsager = deepcopy(self.B_U)
        self.A_U_onsager = deepcopy(self.A_U)
        # Update
        (self.U_hat, self.C_hat_U) = self.update_Uhat_Chat_U()
        (self.V_hat, self.C_hat_V) = self.update_Vhat_Chat_V()
        if self.with_structured_prior:
            (self.W_hat, self.C_hat_W) = self.update_What_Chat_W()

        self.compute_overlap()

        self.break_AMP, self.m_q_close_bayes = False, False
        self.step, self.diff = 0, 0
        self.threshold_error = self.threshold_error_overlap if self.diff_mode == 'overlap' else self.threshold_error_weight
        self.diff = self.threshold_error * 10
        self.list_diff = []

    ## Initialization Layer 1, U_hat C_hat_U, V_hat C_hat_V
    def initialization_Uhat_Chat_U(self):
        """
        Initialization Layer 1, U_hat C_hat_U
        """
        U_hat = np.zeros((self.K, self.M))
        C_hat_U = np.zeros((self.K, self.K, self.M))
        for mu in range(self.M):
            U_hat[:, mu] = self.initialization_Uhat(mu)
            C_hat_U[:, :, mu] = self.initialization_Chat_U()
        self.U_hat = U_hat
        self.C_hat_U = C_hat_U

    def initialization_Uhat(self, mu):
        """
        Initializes U_hat
        - average : uses prior mean
        - random : random vector
        - planted : ground truth + noise
        """
        if self.initialization_mode == 'average':
            U_hat = np.zeros(self.K)
        elif self.initialization_mode == 'random':
            U_hat = self.coef_initialization_random * np.random.random(self.K)
        elif self.initialization_mode == 'planted':
            U_hat = self.U[:, mu] + self.planted_noise * \
                np.random.random(self.K)
        else:
            raise NameError('Wrong intialization U_hat')
        return U_hat

    def initialization_Chat_U(self):
        """
        Initializes C_hat_U with variance of the prior on U
        """
        C_hat_U = np.identity(self.K)
        return C_hat_U

    def initialization_B_A_U(self):
        """
        Initializes B_U, A_U
        - mode == 'update_no_Onsager': updating B_U, A_U without the Onsager term
        - mode == ' ' : usign spd matrix and random vector
        """
        mode = 'update_no_Onsager'
        if mode == 'update_no_Onsager':
            self.B_U = self.update_B_U()
            self.A_U = np.abs(self.update_A_U())
        else:
            for mu in range(self.M):
                self.A_U[:, :, mu] = make_spd_matrix(self.K)
            self.B_U = np.random.randn(self.K, self.M)

    def initialization_Vhat_Chat_V(self):
        """
        Initialization Layer 1, V_hat C_hat_V
        """
        if self.model == 'UU':
            self.V_hat = self.U_hat
            self.C_hat_V = self.C_hat_U
        elif self.model == 'VU':
            V_hat = np.zeros((self.K, self.P))
            C_hat_V = np.zeros((self.K, self.K, self.P))
            for j in range(self.P):
                V_hat[:, j] = self.initialization_Vhat(j)
                C_hat_V[:, :, j] = self.initialization_Chat_V()
            self.V_hat = V_hat
            self.C_hat_V = C_hat_V

    def initialization_Vhat(self, j):
        """
        Initializes V_hat
        - average : uses prior mean
        - random : random vector
        - planted : ground truth + noise
        """
        if self.initialization_mode == 'average':
            if self.weights == 'gaussian':
                V_hat = self.lambda_v
            elif self.weights == 'binary':
                V_hat = np.ones(self.K) * 0.1
            else:
                raise NameError('weights not defined')
        elif self.initialization_mode == 'random':
            V_hat = self.coef_initialization_random * np.random.random(self.K)
            #V_hat = np.ones(self.K)
        elif self.initialization_mode == 'planted':
            V_hat = self.W1[:, j] + self.planted_noise * \
                np.random.random(self.K)
        else:
            raise NameError('Wrong intialization V_hat')
        return V_hat

    def initialization_Chat_V(self):
        """
        Initializes C_hat_V with variance of the prior on V
        """
        if self.weights == 'gaussian':
            C_hat_V = self.Sigma_v
        elif self.weights == 'binary':
            C_hat_V = np.identity(self.K)
        else:
            raise NameError('weights not defined')
        return C_hat_V

    def initialization_B_A_V(self):
        """
        Initializes B_V, A_V
        - mode == 'update_no_Onsager': updating B_V, A_V without the Onsager term
        - mode == ' ' : usign spd matrix and random vector
        """
        mode = 'update_no_Onsager'
        if mode == 'update_no_Onsager':
            self.B_V = self.update_B_V()
            self.A_V = self.update_A_V()
        else:
            for j in range(self.P):
                self.A_V[:, :, j] = make_spd_matrix(self.K)
            self.B_V = np.random.randn(self.K, self.P)

    ## Initialization Layer 2, W_hat C_hat_W
    def initialization_What_Chat_W(self):
        """
        Initialization Layer 2, W_hat C_hat_W
        """
        W_hat = np.zeros((self.K, self.N))
        C_hat_W = np.zeros((self.K, self.K, self.N))
        for i in range(self.N):
            W_hat[:, i] = self.initialization_What(i)
            C_hat_W[:, :, i] = self.initialization_Chat_W()
        self.W_hat = W_hat
        self.C_hat_W = C_hat_W

    def initialization_What(self, i):
        """
        Initializes W_hat
        - average : uses prior mean
        - random : random vector
        - planted : ground truth + noise
        """
        if self.initialization_mode == 'average':
            if self.weights == 'gaussian':
                W_hat = self.lambda_w
            elif self.weights == 'binary':
                W_hat = np.ones(self.K) * 0.1
            else:
                raise NameError('weights not defined')
        elif self.initialization_mode == 'random':
            W_hat = self.coef_initialization_random * np.random.random(self.K)
        elif self.initialization_mode == 'planted':
            W_hat = self.W2[:, i] + self.planted_noise * \
                np.random.random(self.K)
        else:
            raise NameError('Wrong intialization W_hat')
        return W_hat

    def initialization_Chat_W(self):
        """
        Initializes C_hat_W with variance of the prior on W
        """
        if self.weights == 'gaussian':
            C_hat_W = self.Sigma_w
        elif self.weights == 'binary':
            C_hat_W = np.identity(self.K)
        else:
            raise NameError('weights not defined')
        return C_hat_W

    def initialization_V_omega(self):
        """
        Initializes V and omega
        - mode == 'update_no_Onsager': updating V, omega without the Onsager term
        - mode == ' ' : usign spd matrix and random vector
        """
        mode = 'update_no_Onsager'
        if mode == 'update_no_Onsager':
            self.V, self.V_inv = self.update_V()
            self.omega = self.update_omega()
        else:
            for mu in range(self.M):
                self.V[:, :, mu] = make_spd_matrix(self.K)
                self.V_inv[:, :, mu] = inv(self.V[:, :, mu])
                self.omega = np.random.randn(self.K, self.M)

    def initialization_gout_dgout(self):
        """
        Initializes gout, dgout as an update
        """
        (self.gout, self.dgout) = self.update_gout_dgout()
        # To avoid problem if initialization of dgout == 0 for sign
        if np.all(self.dgout == 0):
            self.dgout = - np.abs(np.random.random((self.K, self.K, self.M)))

    def initialization_sigma_lambda(self):
        """
        Initializes sigma, lambda as an update
        """
        self.sigma_inv, self.sigma = self.update_sigma()
        self.lambd = self.update_lambda()

    ############## Updates ##############
    ## Update Layer 1, U_hat C_hat_U, V_hat C_hat_V
    def update_Uhat_Chat_U(self):
        """
        Updates U_hat and Chat_U
        """
        U_hat = np.zeros((self.K, self.M))
        C_hat_U = np.zeros((self.K, self.K, self.M))
        for mu in range(self.M):
            if self.print_gout and (mu/self.M * 100 % 10 == 0):
                print('fu = '+str(mu/self.M*100.)+'%')
            U_hat[:, mu], C_hat_U[:, :, mu] = self.fU_fC(mu)
        return (U_hat, C_hat_U)

    def fU_fC(self, mu):
        """
        Computes fU and fC:
            fU = int du int dz u P_out(u|z) N_u( , ) N_z( , )
            fC = int du int dz u^2 P_out(u|z) N_u( , ) N_z( , ) - fU^2

        if not self.with_structured_prior: remove messages coming from second layer

        Calls annex classes:
        - fu_0_1_2_relu
        - fu_1_2_sign
        - fu_0_1_2_MC
        """
        Delta_2 = self.Delta_2_mat
        B_U, A_U = self.B_U[:, mu], self.A_U[:, :, mu]
        V, omega = self.V[:, :, mu], self.omega[:, mu]
        if not self.with_structured_prior:
            Delta_2 = 0
            V, omega = self.Sigma_u, np.zeros(self.K)
        
        if self.initialization_no_Onsager_terms:
            B_U, inv_A_U = np.zeros(self.K), np.zeros((self.K, self.K))
        else:
            inv_A_U = inv(A_U)

        # Linear case
        if self.non_linearity_2 == 'linear':
            X = inv(Delta_2 + V)
            fC = inv(A_U + X)
            fU = fC.dot(B_U + X.dot(omega))
        # Other non linearities
        else:
            A = A_U
            invA_B = inv_A_U.dot(B_U)
            # For K=1 with explicit expressions
            if self.K == 1 and self.method_gout == 'explicit':
                if self.non_linearity_2 == 'relu':
                    fu_1_2_relu = fu_0_1_2_relu(
                        Delta_2=Delta_2, invA_B=invA_B, A=A, omega=omega, V=V, inv_A=inv_A_U)
                    fU, fC = fu_1_2_relu.compute_fu_1_2()
                elif self.non_linearity_2 == 'sign':
                    fu_1_2_sign = fu_0_1_2_sign(
                        Delta_2=Delta_2, invA_B=invA_B, A=A, omega=omega, V=V, inv_A=inv_A_U)
                    fU, fC = fu_1_2_sign.compute_fu_1_2()
                else:
                    raise NameError('Method_fu undefined')
            elif self.K == 2 and self.method_gout == 'explicit':
                if self.non_linearity_2 == 'sign':
                    fu_1_2_relu = fu_0_1_2_sign_K2(K=self.K,
                                                   Delta_2=Delta_2, invA_B=invA_B, A=A, omega=omega, V=V, inv_A=inv_A_U)
                    fU, fC = fu_1_2_relu.compute_fu_1_2()
                else:
                    raise NameError('Method_fu undefined')
            # For K>1 with MC simulations
            else:
                if self.non_linearity_2 == 'relu' or self.non_linearity_2 == 'sign':
                    fu_1_2_MC = fu_0_1_2_MC(K=self.K, Delta_2=Delta_2, invA_B=invA_B, A=A,
                                            omega=omega, V=V, inv_A=inv_A_U, non_linearity_2=self.non_linearity_2, N_iter_MCquad=self.N_iter_MC, NProcs=self.NProcs_MC)
                    fU, fC = fu_1_2_MC.compute_fu_1_2()

                else:
                    raise NameError('Method_fu undefined')
        return fU, fC

    def update_B_U(self):
        """
        Updates B_U
        If self.scaling == 'stpr': 1/sqrt(M) Else 1/sqrt(N)
        """
        N = self.M if self.scaling == 'stpr' else self.N
        B_U = 1/sqrt(N) * np.einsum('jm,kj->km', self.S, self.V_hat)
        if not self.initialization_no_Onsager_terms:
            B_U -= 1/N * np.einsum(
                'klm,lm->km', np.einsum('jm,klj->klm', self.S_square, self.C_hat_V), self.U_hat_onsager)
        return B_U

    def update_A_U(self):
        """
        Updates A_U
        If self.scaling == 'stpr': 1/sqrt(M) Else 1/sqrt(N)
        """
        N = self.M if self.scaling == 'stpr' else self.N
        A_U = 1/N * (np.einsum('jm,klj->klm', self.S_square - self.R, np.einsum(
            'kj,lj->klj', self.V_hat, self.V_hat)) - np.einsum('jm,klj->klm', self.R, self.C_hat_V))
        return A_U

    ## Update Layer 1, U_hat C_hat_U, V_hat C_hat_V
    def update_Vhat_Chat_V(self):
        """
        Updates U_hat and Chat_U
        If self.model == 'UU' : V_hat = U_hat
        """
        if self.model == 'UU':
            V_hat = self.U_hat
            C_hat_V = self.C_hat_U
        elif self.model == 'VU'  :
            V_hat = np.zeros((self.K, self.P))
            C_hat_V = np.zeros((self.K, self.K, self.P))
            for j in range(self.P):
                V_hat[:, j], C_hat_V[:, :, j] = self.fV_fC(j)
        return (V_hat, C_hat_V)

    def fV_fC(self, j):
        """
        Computes fV and fC:
            fV = int dv v P_v(v) N_v( , ) 
            fC = int dv v^2 P_v(v) N_v( , ) - fV^2

        Calls annex classes:
        - fv_fw_binary
        """
        A_V, B_V = self.A_V[:, :, j], self.B_V[:, j]
        (fV_1, fV_2) = np.zeros((self.K)), np.zeros((self.K, self.K))
        if self.weights == 'gaussian':
            sigma_star = inv(inv(self.Sigma_v)+A_V)
            lambda_star = sigma_star.dot(
                inv(self.Sigma_v).dot(self.lambda_v) + B_V)
            fV_1 = lambda_star
            fV_2 = sigma_star
        elif self.weights == 'binary':
            fv_1_2_binary = fv_fw_binary(K=self.K, B=B_V, A=A_V)
            fV_1, fV_2 = fv_1_2_binary.compute_fv_w_1_2()
        else:
            raise NameError('fV_fC not defined for this prior')
        return (fV_1, fV_2)

    def update_B_V(self):
        """
        Updates B_V
        If self.scaling == 'stpr': 1/sqrt(M) Else 1/sqrt(N)
        """
        N = self.M if self.scaling == 'stpr' else self.N
        B_V = 1/sqrt(N) * np.einsum('jm,km->kj', self.S, self.U_hat)
        if not self.initialization_no_Onsager_terms:
            B_V -= 1/N * np.einsum(
                'klj,lj->kj', np.einsum('jm,klm->klj', self.S_square, self.C_hat_U), self.V_hat_onsager)
        return B_V

    def update_A_V(self):
        """
        Updates A_V
        If self.scaling == 'stpr': 1/sqrt(M) Else 1/sqrt(N)
        """
        N = self.M if self.scaling == 'stpr' else self.N
        A_V = 1/N * (np.einsum('jm,klm->klj', self.S_square-self.R, np.einsum(
            'km,lm->klm', self.U_hat, self.U_hat)) - np.einsum('jm,klm->klj', self.R, self.C_hat_U))    
        return A_V

    ## Update Layer 2, W_hat C_hat_W
    def update_What_Chat_W(self):
        """
        Updates W_hat and Chat_W
        """
        W_hat = np.zeros((self.K, self.N))
        C_hat_V = np.zeros((self.K, self.K, self.N))
        for i in range(self.N):
            W_hat[:, i], C_hat_V[:, :, i] = self.fW_fC(i)
        return (W_hat, C_hat_V)

    def fW_fC(self, i):
        """
        Computes fW and fC:
            fW = int dv w P_w(w) N_w( , ) 
            fC = int dv w^2 P_w(w) N_w( , ) - fW^2

        Calls annex classes:
        - fv_fw_binary
        """
        sigma_inv, lambd = self.sigma_inv[:, :, i], self.lambd[:, i]
        (fW_1, fW_2) = np.zeros((self.K)), np.zeros((self.K, self.K))
        if self.weights == 'gaussian':
            sigma_star = inv(inv(self.Sigma_w) + sigma_inv)
            lambda_star = sigma_star.dot(inv(self.Sigma_w).dot(
                self.lambda_w) + sigma_inv.dot(lambd))
            fW_1 = lambda_star
            fW_2 = sigma_star
        elif self.weights == 'binary':
            fw_1_2_binary = fv_fw_binary(
                K=self.K, B=sigma_inv.dot(lambd), A=sigma_inv)
            fW_1, fW_2 = fw_1_2_binary.compute_fv_w_1_2()
        else:
            raise NameError('fW_fC not defined for this prior')
        return (fW_1, fW_2)

    def update_lambda(self):
        """
        Update lambda^{t+1} = \sigma^{t+1} * ( 1/sqrt(N) * X * gout  - 1/N * V * dgout * W_hat) 
        """
        lambd = np.einsum('kli,ki->ki', self.sigma, 1/sqrt(self.N) * np.einsum('im,km->ki', self.X, self.gout) -
                          1/self.N * np.einsum('kli,li->ki', np.einsum('im,klm->kli', self.X_square, self.dgout), self.W_hat))
        return lambd

    def update_sigma(self):
        """
        Update sigma_inv^{t+1} = - 1/N * X * dgout
        """
        sigma_inv = - 1/self.N * \
            np.einsum('im,klm->kli', self.X_square, self.dgout)
        try:
            sigma = np.moveaxis(inv(np.moveaxis(sigma_inv, -1, 0)), 0, -1)
        except:
            raise NameError('Sigma not invertible')
        return sigma_inv, sigma

    def update_omega(self):
        """
        Update omega^{t} = 1/sqrt(N) * X * W_hat - 1/N * X * C_hat_W * gout
        """
        omega = 1/sqrt(self.N) * np.einsum('im,ki ->km', self.X, self.W_hat)
        if not self.initialization_no_Onsager_terms:
            omega -= 1/self.N * np.einsum(
                'klm,lm->km', np.einsum('im,kli->klm', self.X_square, self.C_hat_W), self.gout)
        return omega

    def update_V(self):
        """ 
        Update V and V_inv
        """
        V = 1 / self.N * np.einsum('im,kli ->klm',self.X_square, self.C_hat_W)
        V_inv = np.moveaxis(np.linalg.inv(np.moveaxis(V, -1, 0)), 0, -1)
        return V, V_inv

    def update_gout_dgout(self):
        """
        Update gout dgout
        - if self.averaged_dgout: average dgout
        """
        gout = np.zeros((self.K, self.M))
        dgout = np.zeros((self.K, self.K, self.M))
        for m in range(self.M):
            if self.print_gout and (m/self.M * 100 % 10 == 0):
                self.print_('gout = {m/self.M*100.:.0f}%')
            gout[:, m], dgout[:, :, m] = self.compute_gout_dgout(m)

        # Average dgout
        if self.averaged_dgout:
            dgout_avg = np.mean(dgout, axis=2)
            for mu in range(self.M):
                dgout[:, :, mu] = dgout_avg
        return (gout, dgout)

    def compute_gout_dgout(self, mu):
        """
        Computes gout dgout:
            gout = V_inv * int du int dz [z-omega] P_out(u|z) N_u( , ) N_z( , )
            dgout = V_inv^2 int du int dz [z-omega]^2 P_out(u|z) N_u( , ) N_z( , ) - V_inv - gout^2

        if not self.with_structured_prior: remove messages coming from second layer

        Calls annex classes:
        - fout_gout_relu
        - fout_gout_sign
        - fout_gout_MC
        """
        Delta_2 = self.Delta_2_mat
        V, omega = self.V[:, :, mu], self.omega[:, mu]
        #B_U, A_U = self.B_U_onsager[:, mu], self.A_U_onsager[:, :, mu]
        B_U, A_U = self.B_U[:, mu], self.A_U[:, :, mu]
        if self.initialization_no_Onsager_terms:
            B_U, inv_A_U = np.zeros(self.K), np.zeros((self.K, self.K))
        else:
            inv_A_U = inv(A_U)

        # Linear case
        if self.non_linearity_2 == 'linear':
            (gout, dgout) = self.compute_gout_dgout_linear(
                B_U, inv_A_U, omega, V, Delta_2)
        # Other non linearities
        else:
            A = A_U
            invA_B = inv_A_U.dot(B_U)
            # For K=1 with explicit expressions
            if self.K == 1 and self.method_gout == 'explicit':
                if self.non_linearity_2 == 'relu':
                    fout_gout_relu = fout_gout_dgout_relu(
                        Delta_2=Delta_2, invA_B=invA_B, A=A, omega=omega, V=V, inv_A=inv_A_U, Nishimori_identity=self.Nishimori_identity)
                    (gout, dgout) = fout_gout_relu.compute_gout_dgout()
                elif self.non_linearity_2 == 'sign':
                    fout_gout_sign = fout_gout_dgout_sign(K=self.K,
                                                          Delta_2=Delta_2, invA_B=invA_B, A=A, omega=omega, V=V, inv_A=inv_A_U, Nishimori_identity=self.Nishimori_identity)
                    (gout, dgout) = fout_gout_sign.compute_gout_dgout()
                else:
                    raise NameError('Method_gout undefined')
            elif self.K == 2 and self.method_gout == 'explicit':
                if self.non_linearity_2 == 'sign':
                    fout_gout_sign = fout_gout_dgout_sign_K2(K=self.K,
                                                             Delta_2=Delta_2, invA_B=invA_B, A=A, omega=omega, V=V, inv_A=inv_A_U)
                    (gout, dgout) = fout_gout_sign.compute_gout_dgout()
                else:
                    raise NameError('Method_gout undefined')
            # For K>1 with MC simulations
            else:
                if self.non_linearity_2 == 'relu' or self.non_linearity_2 == 'sign':
                    fout_gout_MC = fout_gout_dgout_MC(K=self.K,
                                                      Delta_2=self.Delta_2, invA_B=invA_B, A=A, omega=omega, V=V, inv_A=inv_A_U, Nishimori_identity=self.Nishimori_identity, non_linearity_2=self.non_linearity_2, N_iter_MCquad=self.N_iter_MC, NProcs=self.NProcs_MC)
                    (gout, dgout) = fout_gout_MC.compute_gout_dgout()
                else:
                    raise NameError('Method_gout undefined')

        return gout, dgout

    def compute_gout_dgout_linear(self, B_U, inv_A_U, omega, V, Delta_2):
        """
        Compute gout, dgout for linear channel
        """
        X = inv(inv_A_U + Delta_2 + V)
        gout = X.dot(inv_A_U.dot(B_U) - omega)
        dgout = - X
        return (gout, dgout)

    ############## Annex ##############
    def compute_overlap(self):
        """ 
        Compute overlap parameters
        q_X = 1/size * X' X
        m_X = 1/size * X' X0
        """
        if self.model == 'UU':
            m_v = 1/self.M * np.abs(self.V_hat.dot(self.U.T))
            q_v = 1/self.M * self.V_hat.dot(self.V_hat.T)
        elif self.model == 'VU':
            m_v = 1/self.P * np.abs(self.V_hat.dot(self.W1.T))
            q_v = 1/self.P * self.V_hat.dot(self.V_hat.T)
        m_w = 1/self.N * np.abs(self.W_hat.dot(self.W2.T))
        q_w = 1/self.N * self.W_hat.dot(self.W_hat.T)
        m_u = 1/self.M * np.abs(self.U_hat.dot(self.U.T))
        q_u = 1/self.M * self.U_hat.dot(self.U_hat.T)
        # Store old overlaps
        self.m_v_old, self.q_v_old, self.m_u_old, self.q_u_old, self.m_w_old, self.q_w_old = self.m_v, self.q_v, self.m_u, self.q_u, self.m_w, self.q_w
        # Update them
        self.m_v, self.q_v, self.m_u, self.q_u, self.m_w, self.q_w = m_v, q_v, m_u, q_u, m_w, q_w                                                                                 
        list_m_q = [self.m_u, self.q_u] 
        self.list_evolution_m_q.append(list_m_q)
        self.tab_m_q = np.array(list_m_q)
        return self.tab_m_q

    def compute_MSE(self):
        """ 
        Compute the MSE in Bayes Optimal setting
        MSE = Q^0_X - q_X
        """
        #MSE_v, MSE_u, MSE_w = 1 + self.q_v - 2 *self.m_v  , 1 + self.q_u - 2 *self.m_u, 1 + self.q_w - 2 *self.m_w
        MSE_v, MSE_w = 1 - self.m_v, 1 - self.m_w
        if self.non_linearity_2 == 'relu':
            MSE_u = 1/2 - self.m_u
        else:
            MSE_u = 1 - self.m_u
        self.MSE_v = MSE_v
        self.MSE_u = MSE_u
        self.MSE_w = MSE_w
        self.tab_MSE = [MSE_u]
        print(f'm_v = {self.m_u[0][0]:.3f} q_v = {self.q_u[0][0]:.3f} MSE_v = {self.MSE_u.item() :.3f}') if self.verbose else 0

    def compute_difference(self):
        """ 
        Compute difference between t and t+1 of :
        - overlaps | q_X^{t} - q_X^{t+1} |
        - weights  | X^{t} - X^{t+1} |
        - Bayes optimal | m^{t} - q^{t+1} |
        """
        diff_weights = 1/self.P * np.sum([norm(self.V_hat[:, j] - self.V_hat_onsager[:, j]) for j in range(
            self.P)]) + 1/self.N * np.sum([norm(self.W_hat[:, i] - self.W_hat_onsager[:, i]) for i in range(self.N)])
        if self.K == 1:
            diff_overlaps = max([np.abs(
                self.m_v-self.m_v_old), np.abs(self.m_u-self.m_u_old), np.abs(self.m_w-self.m_w_old)])
            diff_bayes = max(
                np.abs(np.array([self.m_v-self.q_v, self.m_u-self.q_u, self.m_w-self.q_w])))
        else:
            diff_overlaps = max([np.abs(norm(self.m_v-self.m_v_old)), np.abs(
                norm(self.m_u-self.m_u_old)), np.abs(norm(self.m_w-self.m_w_old))]) / (self.K**2)
            diff_bayes = max(np.abs(np.array(
                [norm(self.m_v-self.q_v), norm(self.m_u-self.q_u), norm(self.m_w-self.q_w)]))) / (self.K**2)
        if diff_bayes < self.threshold_bayes:
            self.m_q_close_bayes = True
        else:
            self.m_q_close_bayes = False
        diff_ = diff_overlaps if self.diff_mode == 'overlap' else diff_weights
        self.diff_bayes = diff_bayes
        if self.step > self.min_step_AMP:
                self.diff = diff_
        self.list_diff.append(diff_)
        print(f'Diff = {diff_.item():.4e} Diff_bayes = {diff_bayes.item():.4e} / {self.m_q_close_bayes}') if self.verbose else 0

    def to_save(self):
        """
        Remove arrays to save the object
        """
        self.X, self.X_square, self.Y, self.S, self.S_square, self.R = None, None, None, None, None, None
        self.B_V, self.A_V = None, None
        self.B_U, self.A_U = None, None
        self.U_hat, self.C_hat_U = None, None
        self.V_hat, self.C_hat_V = None, None
        self.W_hat, self.C_hat_W = None, None
        self.sigma, self.sigma_inv, self.lambd, self.V, self.V_inv, self.omega, self.gout, self.dgout = None, None, None, None, None, None, None, None
        self.W_hat_onsager, self.V_hat_onsager, self.U_hat_onsager, self.B_U_onsager, self.A_U_onsager = None, None, None, None, None

    ############## Generalization ##############
    def compute_gen_error(self):
        gen_error, gen_error_push = 0, 0
        Delta_1, Delta_2 = self.Delta_1, self.Delta_2
        self.Delta_1, self.Delta_2 = 0, 0
        print('Computing the generalization error...')
        for i in range(self.number_batch_test):
            X_test, Y_test = self.generate_test_set(self.batch_size_test)
            Y_hat = self.Y_hat_optimal(X_test)
            Y_hat_push = self.Y_hat_push_forward(X_test)
            Delta_Y = Y_hat - Y_test
            Delta_Y_push = Y_hat_push - Y_test
            delta_gen_error = np.einsum('km,km', Delta_Y, Delta_Y)
            delta_gen_error_push = np.einsum(
                'km,km', Delta_Y_push, Delta_Y_push)
            gen_error += delta_gen_error
            gen_error_push += delta_gen_error_push
            print('batch', i+1, '/', self.number_batch_test,
                  'intermediate gen:', delta_gen_error/(2*self.batch_size_test))
            print('batch', i+1, '/', self.number_batch_test,
                  'intermediate gen push:', delta_gen_error_push/(2*self.batch_size_test))
        gen_error /= (2 * self.M_test)
        gen_error_push /= (2 * self.M_test)
        print('gen_error =', gen_error)
        self.gen_error = gen_error
        self.gen_error_push_forward = gen_error_push
        self.Delta_1, self.Delta_2 = Delta_1, Delta_2

    def Y_hat_push_forward(self, X_new):
        N, M = X_new.shape
        Z = self.W_hat.dot(X_new) / sqrt(self.N)
        U = self.Phi_out_2(Z).reshape(self.K, M)
        Z1 = (self.V_hat.transpose()).dot(U) / sqrt(self.N)
        Y_hat = self.Phi_out_1(Z1)
        return Y_hat

    def Y_hat_optimal(self, X_new):
        n, m = X_new.shape
        Y_hat = np.zeros((self.P, m))
        for mu in range(m):
            omega = 1/sqrt(self.N) * np.einsum('i,ki ->k',
                                               X_new[:, mu], self.W_hat)  # Omega_new
            V = np.identity(self.K) - self.q_w  # V_new

            if self.non_linearity_2 == 'linear':
                fU = omega
            elif self.non_linearity_2 == 'relu':
                assert self.K == 1
                fU = gaussian(omega, 0, V) * V + \
                    (1+erf(omega/sqrt(2*V))) * 1/2 * omega
            elif self.non_linearity_2 == 'sign':
                assert self.K == 1
                fU = erf(omega/sqrt(2*V))
            else:
                raise NameError('fU undefined')
            fU = fU.reshape(self.K)

            Y_hat[:, mu] = 1 / sqrt(self.N) * (self.V_hat.T).dot(fU)
        return Y_hat

    ############## AMP training ##############
    def check_break_AMP(self):
        """
        Reasons to break AMP iterations
        - cond1: takes too long
        - cond2: preicsion low enough
        - cond3: if rate slow
        """
        ## If takes too long
        cond_1 = self.step > self.max_step_AMP
        ## If precision high enough and reaches Bayes optimality q=m
        cond_2 = self.diff < self.threshold_error and self.m_q_close_bayes
        ## If convergence rate becomes very slow
        n = 10
        if self.step > n:
            rate = np.abs(
                self.list_diff[-1] - self.list_diff[-n]) / (self.list_diff[-n] * n) * 100
            cond_3 = self.step > self.min_step_AMP and rate < 1
        else :
            cond_3 = False
        list_cond = [cond_1, cond_2, cond_3 ]
        if any(list_cond):
            self.break_AMP = True
            #print(f'Breaking conditions: {list_cond}')
        
    def AMP_step(self):
        """ 
        One step of AMP iteration
        """
        self.step += 1
        ## Layer Matrix factorization
        # A_V(t) <- requires: U_hat(t), C_hat_U(t)
        A_V = self.update_A_V()
        self.A_V = self.damping(A_V, self.A_V)
        # B_V(t) <- requires: U_hat(t), C_hat_U(t), V_hat(t-1)
        B_V = self.update_B_V()
        self.B_V = self.damping(B_V, self.B_V)
        # A_U(t) <- requires: V_hat(t), C_hat_V(t)
        A_U = self.update_A_U()
        self.A_U_onsager = deepcopy(self.A_U)
        self.A_U = self.damping(A_U, self.A_U)
        # B_U(t) <- requires: V_hat(t), C_hat_V(t), U_hat(t-1)
        B_U = self.update_B_U()
        self.B_U_onsager = deepcopy(self.B_U)
        self.B_U = self.damping(B_U, self.B_U)

        # Onsager
        self.U_hat_onsager = deepcopy(self.U_hat)
        self.V_hat_onsager = deepcopy(self.V_hat)

        ## Layer GLM
        if self.with_structured_prior:
            # V(t) <- requires: C_hat_w(t)
            V, V_inv = self.update_V()
            self.V = self.damping(V, self.V)
            self.V_inv = np.moveaxis(
                inv(np.moveaxis(self.V, -1, 0)), 0, -1) if self.damping_activated else V_inv
            # omega(t) <- requires: w_hat(t),  C_hat_w(t), gout(t-1)
            omega = self.update_omega()
            self.omega = self.damping(omega, self.omega)
            # gout dgout (t) <- requires: B_U(t-1), A_U(t-1), omega(t), sigma(t)
            gout, dgout = self.update_gout_dgout()
            self.gout = self.damping(gout, self.gout)
            self.dgout = self.damping( dgout, self.dgout)
            # sigma, sigma_inv (t)<- requires: dgout(t)
            sigma_inv, sigma = self.update_sigma()
            self.sigma_inv = self.damping(sigma_inv, self.sigma_inv)
            self.sigma = np.moveaxis(inv(np.moveaxis(
                self.sigma_inv, -1, 0)), 0, -1) if self.damping_activated else sigma
            # lambda(t) <- requires: gout(t), dgout(t)
            lambd = self.update_lambda()
            self.lambd = self.damping(lambd, self.lambd)

            # Onsager
            self.W_hat_onsager = deepcopy(self.W_hat)

            # Update of W_hat(t+1), C_hat(t+1) <- needs lambda_t, sigma_t
            (self.W_hat, self.C_hat_W) = self.update_What_Chat_W()

        # Update U_hat(t+1) <- requires: B_U(t), A_U(t), omega(t), V(t)
        (self.U_hat, self.C_hat_U) = self.update_Uhat_Chat_U()
        # Update V_hat(t+1) <- requires: B_V(t), A_V(t)
        (self.V_hat, self.C_hat_V) = self.update_Vhat_Chat_V()
       
    def AMP_training(self):
        """
        Iterates AMP as long conditions (check_break_AMP) are False
        """
        while not self.break_AMP:
            print(f'Step = {self.step}') if self.verbose else 0
            self.AMP_step()
            self.compute_overlap()
            self.compute_MSE()
            self.compute_difference()
            self.check_break_AMP()

        ## Post processing ##
        print(f'\n') if self.verbose else 0
        self.verbose = True
        self.compute_overlap()
        self.compute_MSE()
        if self.K == 1 and self.scaling == 'nn':
            self.compute_gen_error()

    def main(self):
        """
        MAIN:
        - Initialiation storage
        - Generate training set
        - Initialization
        - Training
        """
        self.initialization_storage()
        self.generate_training_set()
        self.initialization()
        self.AMP_training()


class AMP_averaged(object):
    def __init__(self, K=1, N=1000, alpha_1=1, alpha_2=1, non_linearity_1='linear', non_linearity_2='linear', weights='gaussian', Delta_1=0.01, Delta_2=0.01, method_gout='explicit', seed=False, save=True, N_average=1, initialization_mode='planted', model='VU', verbose='True'):
        print(f'Model: {model}')
        if model == 'VV':
            model ='UU'
        elif model == 'UV':
            model = 'VU'
        else : 
            raise Exception('Undefined model')
        self.K = K
        self.N = N
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.P = int(self.N * self.alpha_1 * self.alpha_2)
        self.M = int(self.N * self.alpha_2)
        self.non_linearity_1 = non_linearity_1
        self.non_linearity_2 = non_linearity_2
        self.weights = weights
        self.Delta_1 = Delta_1
        self.Delta_2 = Delta_2
        self.seed = seed
        self.method_gout = method_gout
        self.initialization_mode = initialization_mode
        self.model = model
        self.verbose = verbose
        
        
        print(f'k: {self.N} p: {self.M} n: {self.P}')
        print(f"beta: {self.alpha_1} alpha: {self.alpha_2}")
        print(f"non_linearity: {self.non_linearity_2}")
        print(f"Delta: {self.Delta_1}")
        print(f"Initialization: {self.initialization_mode}")

        self.save = save
        self.N_average = N_average
        self.average_m_q = [0, 0, 0, 0, 0, 0]
        self.std_m_q = [0, 0, 0, 0, 0, 0]
        self.average_MSE = [0, 0, 0]
        self.std_MSE = [0, 0, 0]
        self.average_gen_error = 0
        self.std_gen_error = 0
        self.average_gen_error_push_forward = 0
        self.std_gen_error_push_forward = 0
        self.list_average_m_q = []
        self.list_average_MSE = []
        self.list_average_gen_error = []
        self.list_average_gen_error_push_forward = []

    def main(self):
        self.step = 1
        for i in range(self.N_average):
            self.single_average(i)
            self.step += 1

    def single_average(self, i):
        print(f'\nIteration: {i+1} / {self.N_average}') if self.verbose else 0
        AMP_ = AMP(K=self.K,  N=self.N, alpha_1=self.alpha_1, alpha_2=self.alpha_2, non_linearity_1=self.non_linearity_1,
                   non_linearity_2=self.non_linearity_2, weights=self.weights, Delta_1=self.Delta_1, Delta_2=self.Delta_2, seed=self.seed,    method_gout=self.method_gout, initialization_mode=self.initialization_mode, model=self.model, verbose=self.verbose)
        AMP_.main()
        self.list_average_m_q.append(AMP_.tab_m_q)
        self.list_average_MSE.append(AMP_.tab_MSE)
        self.list_average_gen_error.append(AMP_.gen_error)
        self.list_average_gen_error_push_forward.append(
            AMP_.gen_error_push_forward)
        self.average()
        self.copy_save(AMP_)

    def average(self):
        # Average overlaps
        self.average_m_q = np.mean(
            np.array(self.list_average_m_q), axis=0).T[0][0]
        self.std_m_q = np.std(np.array(self.list_average_m_q), axis=0).T[0][0]
        print(
            f'm_v avg = {self.average_m_q[0]:.3f} q_v avg = {self.average_m_q[1]:.3f} m_v std = {self.std_m_q[0]:.3f} q_v std = {self.std_m_q[1]:.3f}') if self.step == self.N_average else 0
        self.q_v = self.average_m_q[1]

        # Average MSE
        self.average_MSE = np.mean(np.array(self.list_average_MSE), axis=0).T[0][0]
        self.std_MSE = np.std(np.array(self.list_average_MSE), axis=0).T[0][0]
        print(
            f'MSE_v avg = {self.average_MSE[0]:.3f} MSE_v std = {self.std_MSE[0]:.3f}') if self.step == self.N_average else 0
        self.MSE_v = self.average_MSE[0]

        # Average Gen error
        self.average_gen_error = np.mean(
            np.array(self.list_average_gen_error), axis=0)
        self.average_gen_error_push_forward = np.mean(
            np.array(self.list_average_gen_error_push_forward), axis=0)
        self.std_gen_error_push_forward = np.std(
            np.array(self.list_average_gen_error_push_forward), axis=0)
        #print('gen_error avg push forward :', self.average_gen_error_push_forward,
         #     ' gen_error std push forward:', self.std_gen_error_push_forward)

    def copy_save(self, obj):
        obj.to_save()
        obj.N_average = self.N_average
        obj.list_average_m_q = self.list_average_m_q
        obj.average_m_q = self.average_m_q
        obj.std_m_q = self.std_m_q
        obj.list_average_MSE = self.list_average_MSE
        obj.average_MSE = self.average_MSE
        obj.std_MSE = self.std_MSE
        obj.average_gen_error = self.average_gen_error
        obj.std_gen_error = self.std_gen_error

        #obj.q_v, obj.q_u, obj.q_w = self.m_v, self.m_u, self.m_w
        #obj.MSE_v, obj.MSE_u, obj.MSE_w = self.MSE_v, self.MSE_u, self.MSE_w
        if self.save:
            file = obj.file_name
            save_object(obj, file)
