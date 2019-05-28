from Library.import_library import *
from Functions.functions_save_load import save_object, load_object
from Functions.generic_functions import sign, relu
from Class.fout_fu_relu import fout_gout_dgout_relu, fu_0_1_2_relu
from Class.fout_fu_sign import fout_gout_dgout_sign, fu_0_1_2_sign

class Spectral(object):
    def __init__(self, K = 1, N = 1000, alpha_1 = 1, alpha_2 = 1, non_linearity_1 = 'linear', non_linearity_2 = 'linear', weights = 'gaussian', Delta_1 = 0.01, Delta_2 = 0.01, method_gout='explicit', seed=False, model = 'VU', verbose ='True'):
        ## Scaling 
        self.scaling = 'stpr'
        ## Parameters
        self.K = K
        self.N = N
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2
        self.model = model
        ## Limit alpha_2 = 0 : no prior
        self.with_structured_prior = True
        if self.alpha_2 == 0:
            self.with_structured_prior = False
            self.alpha_2 = 1 
        ## Change definition of P depending on the scaling
        if self.scaling == 'nn':
            self.P = int(self.N * self.alpha_1)
        elif self.scaling == 'stpr':
            self.P = int(self.N * self.alpha_1 * self.alpha_2)
        self.M = int(self.N * self.alpha_2)
        self.verbose = verbose
        
        self.non_linearity_1 = non_linearity_1
        self.non_linearity_2 = non_linearity_2
        self.Delta_1, self.Delta_2 = Delta_1, Delta_2
        self.method = 'eigs'
        # if self.verbose:
        #     print(f'Model: {self.model}')
        #     print(f'Scaling: {self.scaling}')
        #     print(f'K: {self.K} N: {self.N} M:{self.M} P:{self.P}')
        #     print(f"alpha_1: {self.alpha_1} alpha_2: {self.alpha_2}")
        #     print(
        #         f"non_linearity_1: {self.non_linearity_1} non_linearity_2: {self.non_linearity_2}")
        #     print(f"Delta_1: {self.Delta_1} Delta_2: {self.Delta_2}")
        #     print(f"Method eivp: {self.method}")

        ## Test set
        self.number_batch_test, self.batch_size_test = 10, 10000
        self.M_test = self.number_batch_test * self.batch_size_test  # size test_set

        ## Mean and covariance
        self.weights = weights
        self.lambda_v, self.lambda_z, self.lambda_u = np.zeros(
            (self.K)), np.zeros((self.K)), np.zeros((self.K))  # mean for gaussian
        self.Sigma_v, self.Sigma_z, self.Sigma_u = np.identity(
            self.K), np.identity(self.K), np.identity(self.K)  # covariance for gaussian

		## Data: gaussian with zero mean and variance 1
        self.lambda_X, self.Sigma_X = 0, 1

		### Average overlaps
        self.N_average = 0
        self.gen_error = 0

        ## Mode
        self.model = model

        ## Seed
        self.seed = seed
        np.random.seed(11) if self.seed else 0
        #print(f'Seed {self.seed}')
            
        self.print_Initialization = False
		## Plot Options
        self.plot_Fig, self.save_Fig = False, False

        ## Directory
        self.path_Fig = 'Figures/'
        if not os.path.exists(self.path_Fig):
            os.makedirs(self.path_Fig)

        ## Spectral
        self.directory = self.non_linearity_1 + '_' + self.non_linearity_2 + \
            '_' + 'K=' + str(self.K) + '/'
        self.data_directory = 'Data/Spectral_' + model +'/'+ self.directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        self.name = 'Spectral_SeqToSeq_K='+str(self.K)+'_a1=' + '%.4f' % self.alpha_1+'_a2='+'%.4f' % self.alpha_2 + '_d1=' + '%.6f' % self.Delta_1 + \
            '_d2='+'%.6f' % self.Delta_2 + '_'+self.non_linearity_1 + '_' + \
            self.non_linearity_2 + '_' + self.weights + '_N='+str(self.N) + '_'+self.model
        self.file_name = self.data_directory + self.name + '.pkl'


        ## Other spectral methods
        self.tab_method = ['PCA', 'lAMP'] 
        self.tab_m_q_other_methods = []
        self.tab_MSE_other_methods = []

        self.generate_parameters()
        ##########################
    
    def generate_parameters(self):
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
        self.dict_parameters['weights'] = self.weights
        self.dict_parameters['seed'] = self.seed
        self.dict_parameters['N_average'] = self.N_average
        self.dict_parameters['file_name'] = self.file_name
        #print(self.dict_parameters)
    
    def to_save(self):
        self.X, self.Y = None, None
        self.U, self.V, self.Z = None, None, None

    def initialization_storage(self):
        self.X = np.zeros((self.N,self.M))
        self.Y = np.zeros((self.P,self.M))
        self.Z = np.zeros((self.K,self.N))
        self.V = np.zeros((self.K,self.P))
        self.U = np.zeros((self.K,self.M))
        self.list_evolution_m_q = []
        self.m_v, self.q_v, self.m_u, self.q_u, self.m_w, self.q_w = 0, 0, 0, 0, 0, 0
        self.tab_m_q, self.tab_MSE = [], []
    ############## Generate data ##################
    # Generate data set X
    def generate_X(self, M_samples):
        X = np.random.normal(self.lambda_X, self.Sigma_X, (M_samples, self.N))
        if self.print_Initialization:
            print('X=', X, '\n')
        return X
    
    # Generate teacher weights U, V
    def generate_U_V_Z(self):
        # Gaussian
        if self.weights == 'gaussian':
            for j in range(self.P):
               self.V[:,j] = np.random.multivariate_normal(self.lambda_v, self.Sigma_v)
            for mu in range(self.M):
               self.U[:,mu] = np.random.multivariate_normal(self.lambda_u, self.Sigma_u)
            for i in range(self.N):
               self.Z[:, i] = np.random.multivariate_normal(
                   self.lambda_z, self.Sigma_z)
        else :
            raise NameError('Weights Prior weights not defined')
        return (self.V , self.U)

    # Phi_out_1, Phi_out_2
    def Phi_out_1(self, z1):
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
            resul += (noise+noise.T)/sqrt(2)
        else : 
            resul += noise
        return resul

    def Phi_out_2(self, z2):
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
    
    # Generate teacher labels
    def generate_Y(self,X):
        if self.model == 'UU':
            Y, U = self.generate_Y_UU(X)
        elif self.model =='VU':
            Y, U = self.generate_Y_VU(X)
        return (Y, U)

    def generate_Y_UU(self, X):
        if self.alpha_2 == 0:
            U = self.U
        else:
            Z = (X.dot(self.Z.T) / sqrt(self.N))
            U = self.Phi_out_2(Z)
        if self.scaling == 'stpr':
            Y = self.Phi_out_1(U.dot(U.T) / sqrt(self.M))
        elif self.scaling == 'nn':
            Y = self.Phi_out_1(U.dot(U.T) / sqrt(self.N))
        return (Y, U)

    def generate_Y_VU(self, X):
        if self.alpha_2 == 0:
            U = self.U
        else:
            Z = (X.dot(self.Z.T) / sqrt(self.N))
            U = self.Phi_out_2(Z)
        if self.scaling == 'stpr':
            Y = self.Phi_out_1(self.V.T.dot(U.T) / sqrt(self.M))
        elif self.scaling == 'nn':
            Y = self.Phi_out_1(self.V.T.dot(U.T) / sqrt(self.N))
        return (Y, U)

    def generate_training_set(self):
        self.X = self.generate_X(self.M)
        self.generate_U_V_Z()
        self.Y, self.U = self.generate_Y(self.X)

    def generate_test_set(self, M_test):
        X_test = self.generate_X(M_test)
        Y_test, U = self.generate_Y(X_test)
        return X_test, Y_test
        
    ##################### SPECTRAL SVD ######################
    def training(self):
        self.tab_m_q = []
        self.tab_MSE = []
        self.gen_error = 0
        for method in self.tab_method : 
            if self.model == 'UU':
                U_hat = self.training_UU(method)
            elif self.model == 'VU':
                U_hat = self.training_VU(method)
            
            tab_m_q_, tab_MSE_ = self.compute_overlap_MSE(U_hat, method)
            self.tab_m_q.extend(tab_m_q_)
            self.tab_MSE.extend(tab_MSE_)

    def training_UU(self, method):
        if method == 'XXY':
            Gamma = 1/self.N * (self.X.dot(self.X.T)).dot(self.Y) / sqrt(self.M)
        elif method == 'PCA':
            Gamma = self.Y / sqrt(self.M)
        elif method == '(1+XX)Y':
            Gamma = (np.identity(self.M) + 1/self.N * self.X.dot(self.X.T)).dot(self.Y) / sqrt(self.M)
        elif method == 'lAMP':
            if self.non_linearity_2 == 'linear':
                a, b = 1 , 1
            elif self.non_linearity_2 == 'sign':
                a, b = 1, 2/pi
            else:
                a, b = 1, 0
            Sigma = b * self.X.dot(self.X.T) / self.N + (a-b) * np.identity(self.M)
            Gamma = Sigma.dot(self.Y / sqrt(self.M) - a * np.identity(self.M))
        elif method == 'Spectrasson':
            Gamma_yx = 1 / self.M * self.Y.dot(self.X.T)
            _, _, W_hat = randomized_svd(Gamma_yx, self.K)
            U_hat = W_hat.dot(self.X) / sqrt(self.N)
            return U_hat
        else:
            raise NameError('Undefined method')

        U_hat = self.find_eig_sorted(Gamma, method)
        U_hat = U_hat.reshape(self.M, self.K)
        return U_hat

    def training_VU(self, method):
        if method == 'PCA':
            Gamma = self.Y / sqrt(self.M)
            V_hat, D, U_hat = svds(Gamma, k=self.K)
            U_hat = U_hat.reshape(self.M, self.K)
            return U_hat
        elif method == 'lAMP':
            if self.non_linearity_2 == 'linear':
                a, b, d = 1, 1, 1
            elif self.non_linearity_2 == 'sign':
                a, b, d  = 1, 2/pi, 1
            else:
                a, b, d = 1, 1, 1
            Sigma = b * self.X.dot(self.X.T) / self.N + (a-b) * np.identity(self.M)
            Gamma = Sigma.dot(1/(self.Delta_1/d + a) * (self.Y.T).dot(self.Y) / self.M -
                                  d * self.alpha_1 * np.identity(self.M))
        elif method == 'Spectrasson':
            Gamma_xyx = 1/self.N * 1 / \
                sqrt(self.M) * (self.X.T).dot(self.Y.dot(self.X))
            W_hat = self.find_eig_sorted(Gamma_xyx)
            #randomized_svd(Gamma_xyx, self.K)
            W_hat = W_hat.reshape(self.N, self.K)
            self.W_hat = W_hat / norm(W_hat)
            Z = self.X.dot(W_hat) / sqrt(self.N)
            U_hat = Z
            U_hat /= norm(U_hat)
            return U_hat

        else:
            raise NameError(f'Undefined method {method}')

        U_hat = self.find_eig_sorted(Gamma, method)
        U_hat = U_hat.reshape(self.M, self.K)
        return U_hat
    
    def main(self):
        self.tim = time.time()
        self.initialization_storage()
        self.generate_training_set()
        self.training()
        #self.compute_gen_error()
        self.tim = time.time() - self.tim 
        print(f'tim={self.tim}') if self.verbose else 0
        return self.tab_m_q, self.tab_MSE, self.gen_error

    ## Annex functions ##
    def find_eig_sorted(self, L, method=''):
        """
        Returns normalized eigenvector of L wcorresponding to the top real eigenvalue
        """

        n, m = L.shape
        if n != m:
            _, _, wmax = randomized_svd(L, 1)

        else:
            method_ = self.method
            if method_ == 'eig':
                l, w = eig(L)
                index_sorted = np.argsort(l)
                idx = index_sorted[-1]
                lmax = l[idx]
                wmax = w[:, idx]
            elif method_ == 'eigs':
                lmax, wmax = eigs(L, k=1, which='LR')
            elif method_ == 'eigh':
                lmax, wmax = eigh(L, eigvals=(self.M-1, self.M-1))
            else:
                raise NameError('Undefined method')
            lmax = lmax.real
            wmax = wmax.real
        wmax /= norm(wmax)

        return wmax
        
    def compute_gen_error(self):
        gen_error = 0
        print('Computing the generalization error...')
        #Delta_1, Delta_2 = self.Delta_1, self.Delta_2
        #self.Delta_1, self.Delta_2 = 0, 0    
        for i in range(self.number_batch_test):
            X_test, Y_test = self.generate_test_set(self.batch_size_test)
            Y_hat = self.Y_hat_new(X_test)
            Delta_Y = Y_hat - Y_test
            delta_gen_error = np.einsum('km,km', Delta_Y, Delta_Y)
            gen_error += delta_gen_error
            print('batch', i+1, '/', self.number_batch_test,
                  'intermediate gen:', delta_gen_error/(2*self.batch_size_test))
        gen_error /= (2 * self.M_test)
        print('gen_error =', gen_error)
        #self.Delta_1, self.Delta_2 = Delta_1, Delta_2
        self.gen_error = gen_error

    def Y_hat_new(self, X_test):
        N, M = X_test.shape
        Z = self.W_hat.dot(X_test)
        U = self.Phi_out_2(Z).reshape(self.K,M)
        Z1 = (self.V_hat.T).dot(U)
        Y_hat = self.Phi_out_1(Z1)
        return Y_hat

    def compute_overlap_MSE(self, U_hat, method):
        q_u = U_hat.T.dot(U_hat)
        m_u = 1 / sqrt(self.M) * np.abs(U_hat.T.dot(self.U))
        Q = 1/2 if self.non_linearity_2 == 'relu' else 1
        MSE_u = Q + q_u - 2 * m_u
        tab_m_q = [m_u, q_u]
        tab_MSE = MSE_u
        print(f'{method}: m_v = {m_u[0][0]:.3f} q_v = {q_u[0][0]:.3f} MSE_v = {MSE_u[0][0]:.3f}')
        return tab_m_q, tab_MSE 

class Spectral_averaged(object):
    def __init__(self, K=1, N=1000, alpha_1=1, alpha_2=1, non_linearity_1='linear', non_linearity_2='linear', weights='gaussian', Delta_1=0.01, Delta_2=0.01, seed=False, save=True, model='UU', N_average=1, verbose='True'):
        print(f'Model: {model}')
        self.K = K
        self.N = N
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.non_linearity_1 = non_linearity_1
        self.non_linearity_2 = non_linearity_2
        self.weights = weights
        self.Delta_1 = Delta_1
        self.Delta_2 = Delta_2
        self.seed = seed
        if model == 'VV':
            model = 'UU'
        elif model == 'UV':
            model = 'VU'
        else : 
            raise Exception('Undefined model')
        self.model = model
        self.verbose = verbose
        self.P = int(self.N * self.alpha_1 * self.alpha_2)
        self.M = int(self.N * self.alpha_2)
        print(f'k: {self.N} p:{self.M} n:{self.P}')
        print(f"beta: {self.alpha_1} alpha: {self.alpha_2}")
        print(f"non_linearity: {self.non_linearity_2}")
        print(f"Delta: {self.Delta_1}")
        
        self.save = save
        self.N_average = N_average
        self.average_m_q = 0
        self.std_m_q = 0
        self.average_MSE = 0
        self.std_MSE = 0
        self.average_gen_error = 0
        self.std_gen_error = 0

        self.list_average_m_q = []
        self.list_average_MSE = []
        self.list_average_gen_error = []
        
        self.q_v, self.q_u, self.q_w = np.zeros((self.K, self.K)), np.zeros(
            (self.K, self.K)), np.zeros((self.K, self.K))
        self.MSE_v, self.MSE_u, self.MSE_w = np.zeros(
            (1)), np.zeros((1)), np.zeros((1))
        self.gen_error = 0

    def main(self):
        self.step = 1
        for i in range(self.N_average) :
            print('Iteration :'+str(i+1) +'/' +str(self.N_average)) if self.verbose else 0
            spectral = Spectral(K=self.K,  N=self.N, alpha_1=self.alpha_1, alpha_2=self.alpha_2, non_linearity_1=self.non_linearity_1,
                                non_linearity_2=self.non_linearity_2, weights=self.weights, Delta_1=self.Delta_1, Delta_2=self.Delta_2, 
                                seed=self.seed, model=self.model, verbose = self.verbose)
            tab_m_q, tab_MSE, gen_error = spectral.main()
            self.tab_method = spectral.tab_method
            self.list_average_m_q.append(tab_m_q)
            self.list_average_MSE.append(tab_MSE)
            self.list_average_gen_error.append(gen_error)

            self.average()
            self.copy_save(spectral)
            print('\n') if self.verbose else 0
            self.step += 1
        
    def average(self):
        self.average_m_q = np.mean(np.array(self.list_average_m_q),axis=0).T[0][0]
        self.std_m_q = np.std(np.array(self.list_average_m_q), axis=0).T[0][0]
        print(
            f'PCA: m_v avg = {self.average_m_q[0]:.3f} q_v avg = {self.average_m_q[1]:.3f} m_v std = {self.std_m_q[0]:.3f} q_v std = {self.std_m_q[1]:.3f}') if self.step == self.N_average else 0
        print(
            f'lAMP: m_v avg = {self.average_m_q[2]:.3f} q_v avg = {self.average_m_q[3]:.3f} m_v std = {self.std_m_q[2]:.3f} q_v std = {self.std_m_q[3]:.3f}') if self.step == self.N_average else 0

        self.average_MSE = np.mean(np.array(self.list_average_MSE),axis=0).T[0]
        self.std_MSE = np.std(np.array(self.list_average_MSE), axis=0).T[0]
        print(
            f'PCA: MSE_v avg = {self.average_MSE[0]:.3f} MSE_v std = {self.std_MSE[0]:.3f}') if self.step == self.N_average else 0
        print(
            f'lAMP: MSE_v avg = {self.average_MSE[1]:.3f} MSE_v std = {self.std_MSE[1]:.3f}') if self.step == self.N_average else 0

        # self.average_gen_error = np.mean(np.array(self.list_average_gen_error),axis=0)
        # self.std_gen_error = np.std(np.array(self.list_average_gen_error), axis=0)
        # print('gen_error avg :', self.average_gen_error.T,' gen_error std :', self.std_gen_error.T)
        self.q_v_PCA, self.q_v_lAMP = self.average_m_q[0], self.average_m_q[2]
        self.MSE_PCA, self.MSE_lAMP = self.average_MSE[0], self.average_MSE[1]

    def copy_save(self,obj):
        obj.to_save()
        obj.list_average_m_q = self.list_average_m_q
        obj.average_m_q = self.average_m_q
        obj.std_m_q = self.std_m_q
        obj.N_average = self.N_average
        obj.list_average_MSE = self.list_average_MSE
        obj.average_MSE = self.average_MSE
        obj.std_MSE = self.std_MSE
        obj.average_gen_error = self.average_gen_error
        obj.std_gen_error = self.std_gen_error

        obj.q_v, obj.q_u, obj.q_w = self.q_v, self.q_u, self.q_w
        obj.MSE_v, obj.MSE_u, obj.MSE_w = self.MSE_v, self.MSE_u, self.MSE_w
        obj.gen_error = self.gen_error
        
        if self.save :
            file = obj.file_name
            save_object(obj,file)

