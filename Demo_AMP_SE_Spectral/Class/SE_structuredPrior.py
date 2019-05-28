from Library.import_library import *
from Functions.functions_save_load import save_object, load_object
from Functions.generic_functions import sign, relu
from Class.fout_fu_relu import fout_gout_dgout_relu, fu_0_1_2_relu
from Class.fout_fu_sign import fout_gout_dgout_sign, fu_0_1_2_sign



class StateEvolution(object):
    def __init__(self, K=1, alpha_1=1, alpha_2=1, non_linearity_1='linear', non_linearity_2='linear', weights='gaussian', Delta_1=0.01, Delta_2=0.01, method_integration='quad', method_gout='explicit', initialization_mode='informative', model='VU', verbose='True'):
        # Parameters
        self.K = K
        self.alpha_2 = alpha_2
        self.alpha_1 = alpha_1
        self.non_linearity_2 = non_linearity_2
        self.non_linearity_1 = non_linearity_1
        self.weights = weights
        self.Delta_1 = Delta_1
        self.Delta_2 = Delta_2
        self.verbose = verbose
        if self.verbose :
            print(f'K= {self.K}')
            print(f"alpha_1: {self.alpha_1} alpha_2: {self.alpha_2}")
            print(
                f"non_linearity_1: {self.non_linearity_1} non_linearity_2: {self.non_linearity_2}")
            print(f'weights: {self.weights}')
            print(f"Delta_1: {self.Delta_1} Delta_2: {self.Delta_2}")

        # Model
        if model == 'VV':
            model = 'UU'
        elif model == 'UV':
            model = 'VU'
        else :
            raise Exception('Undefined model')
        self.model = model # 'VU' or 'UU'
        print('Model:',self.model) if self.verbose else 0

        # Usefull Matrices
        self.Delta_2_mat = self.Delta_2 * np.identity(self.K)
        self.idty = np.identity(self.K)
        self.Q0_w = self.idty

        # Storage
        self.data = {"q_u": [], "q_v": [], "q_w": [], "q_w_hat": []}
        self.q_v, self.q_u, self.q_w = 0, 0, 0

        # Initialiation
        self.initialization_mode = initialization_mode  # 'informative' or 'random'
        #print('Initialization:', initialization_mode)
        self.initialization_coef_a = 0.1
        self.initialization_coef_d = 0.5

        # Numerical Integration
        self.method_integration = method_integration
        self.method_gout = method_gout
        if self.K == 1:
            # self.method_integration = 'quad'
            self.method_gout = 'explicit'
        #print('Method_gout:', self.method_gout)
        #print('Integration method:', self.method_integration)
        if self.method_integration == 'MC_unif':
            warnings.warn(
                "### Carreful: integration method not very precised ###", Warning)
        if self.method_integration == 'MC_imp':
            warnings.warn(
                "### WARNING: method not working for K>=2 ###", Warning)
        self.borne_sup = 10
        self.borne_inf = -10
        # MC integration of u and xi
        self.N_iter_MC = 100000
        self.N_procs_MC = 4
        # MC integration of fout gout
        self.N_iter_MC_gout = 2000
        self.N_procs_MC_gout = 1
        self.n_ = 0
        self.explicit_integral = True

        # Print
        self.print_gout = False
        if self.K > 1 and self.method_gout == 'MC':
            self.print_gout = True

        # Convergence
        self.step = 0
        self.min_step = 10
        self.tim = time.time()
        self.tim_0 = self.tim
        self.diff = 0

        self.damping_coef = 0.5
        if self.method_integration == 'quad':
            if self.K == 1:
                self.precision = 1e-5
                self.max_steps = 250
            else :
                self.precision = 1e-4
                self.max_steps = 500
        else:
            self.precision = 1e-5
            self.max_steps = 150

        if self.non_linearity_2 == 'linear' and self.K==1 :
            self.max_steps = 10000
            self.precision = 1e-10

        self.structured_prior_mode = True
        if self.structured_prior_mode:
            self.scaling = 'stpr'
        else:
            self.scaling = 'nn'

        # Directory to save
        self.directory = self.non_linearity_1 + '_' + self.non_linearity_2 + \
            '_' + 'K=' + str(self.K) + '/'
        self.data_directory = 'Data/SE_SeqToSeq_'+self.model+'/' + self.directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        # Filename
        self.name = 'SE_SeqToSeq_K='+str(self.K)+'_a1=' + '%.4f' % self.alpha_1+'_a2='+'%.4f' % self.alpha_2 + \
            '_d1=' + '%.6f' % self.Delta_1+'_d2='+'%.6f' % self.Delta_2 + \
            '_'+self.non_linearity_1 + '_'+self.non_linearity_2 + \
            '_' + self.weights + '_'+self.initialization_mode + \
            '_' + self.scaling + '_' + self.method_integration
        self.file_name = self.data_directory + self.name + '.pkl'

    ## Parameters ##
    def def_parameters(self):
        if self.non_linearity_1 == 'linear':
            self.Delta = self.Delta_1
        else:
            raise NameError('Delta not defined for this non-linearity')
        self.Delta = max(self.Delta, 5e-4)
        #print('Delta:', self.Delta)

    ## Initialization + Convergence ##
    def initialization(self):
        if self.initialization_mode == 'informative':
            self.q_w_hat = self.generate_informative_matrix()
            self.q_w = self.generate_informative_matrix()
            self.q_u = self.generate_informative_matrix()
            self.q_v = self.generate_informative_matrix()
        elif self.initialization_mode == 'random':
            self.q_w_hat = self.generate_non_informative_matrix()
            self.q_w = self.generate_non_informative_matrix()
            self.q_u = self.generate_non_informative_matrix()
            self.q_v = self.generate_non_informative_matrix()
        elif self.initialization_mode == 'AMP':
            self.q_v = np.array([[0.50625852, 0.50625852], [
                                0.50625852, 0.50625852]]) + 0.001 * np.identity(self.K)
            self.q_u = np.array([[0.51133677, 0.51133677], [
                                0.51133677, 0.51133677]]) + 0.001 * np.identity(self.K)
            self.q_w = np.array([[0.20003799, 0.20003799], [
                                0.20003799, 0.20003799]]) + 0.001 * np.identity(self.K)
            # 0.5 * np.ones((self.K,self.K)) + 0.001 * np.identity(self.K)
            self.q_w_hat = self.SP_q_w_hat()
            # self.SP_q_w_hat()
        else:
            raise NameError('Undefined Mode')
        if self.model == 'UU':
            self.q_v = self.q_u
        #print(self.q_w_hat, self.q_w, self.q_u, self.q_v)
        self.add_data_to_dict()

    def iteration(self):
        # Single iteration
        q_w = self.SP_q_w()
        q_w_hat = self.SP_q_w_hat()
        q_u = self.SP_q_u()
        if self.model == 'UU':
            q_v = q_u
        else : 
            q_v = self.SP_q_v()
        # Damping
        self.q_w = self.damping(q_w, copy.copy(self.q_w))
        self.q_v = self.damping(q_v, copy.copy(self.q_v))
        self.q_w_hat = self.damping(q_w_hat, copy.copy(self.q_w_hat))
        self.q_u = self.damping(q_u, copy.copy(self.q_u))
        # Add data to dict
        self.add_data_to_dict()
        self.step += 1
        self.tim_ = time.time()
        self.delta_tim = self.tim_ - self.tim
        self.tim = self.tim_
        self.print_last_iteration()

    def difference_overlap(self,str):
        if norm(self.data[str][-1]) > 1e-3 :
            return norm(self.data[str][-1]-self.data[str][-2]) / norm(self.data[str][-1]) 
        else :
            self.precision = 1e-3
            return norm(self.data[str][-1]-self.data[str][-2])
    
    def stopping_criteria(self):
        if self.step < 2:
            return False
        else:
            tab_name = ["q_u", "q_v", "q_w"]
            tab_diff= np.array([ self.difference_overlap(str) for str in tab_name])
            m = max(tab_diff)
            self.diff = m
            if m < self.precision and self.step > self.min_step:
                return True
            else:
                return False

    def main(self):
        self.def_parameters()
        self.initialization()

        while not self.stopping_criteria() and self.step < self.max_steps:
            self.iteration()
        self.q_w, self.q_v, self.q_u, self.q_w_hat = self.data[
            "q_w"][-1], self.data["q_v"][-1], self.data["q_u"][-1], self.data["q_w_hat"][-1]
        self.compute_MSE()
        print(
            f'm_v = {self.q_u[0][0]:.3f} q_v = {self.q_u[0][0]:.3f} MSE_v = {self.MSE_u[0][0]:.3f}')

        #if self.K == 1:
         #   self.compute_gen_error()
          #  print('Gen_error=', self.gen_error)
        
        #print('time:',time.time()-self.tim_0)
        self.q_v = self.q_u[0][0]
        self.MSE_v = self.MSE_u[0][0]

    ## Annex functions ##
    def damping(self, new_obj, old_obj):
        res = new_obj * (1-self.damping_coef) + old_obj * (self.damping_coef)
        return res

    def add_data_to_dict(self):
        self.data["q_u"].append(self.q_u)
        self.data["q_v"].append(self.q_v)
        self.data["q_w"].append(self.q_w)
        self.data["q_w_hat"].append(self.q_w_hat)

    def print_last_iteration(self):
        if self.K == 1 :
            print(
                f'Step: {self.step} Diff:{self.diff:.2e} Time:{self.delta_tim:.0f} q_v:{self.data["q_v"][0][0][-1]:.3f}  \
                q_u:{self.data["q_u"][0][0][-1]:.3f} q_w: {self.data["q_w"][0][0][-1]:.3f} q_w_hat: {self.data["q_w_hat"][0][0][-1]:.3f} ')  if self.verbose else 0
        else:
            print(
                f'Step: {self.step} Diff:{self.diff:.2e} Time:{self.delta_tim:.0f} q_v:{self.data["q_v"][-1]} \
                q_u:{self.data["q_u"][-1]} q_w: {self.data["q_w"][-1]} q_w_hat: {self.data["q_w_hat"][-1]} ') if self.verbose else 0

    def compute_MSE(self):
        if self.non_linearity_2 == 'relu':
            MSE_u =  0.5 - self.q_u
        else :
            MSE_u = 1 - self.q_u
        MSE_v, MSE_w = 1 - self.q_v, 1 - self.q_w
        self.MSE_v = MSE_v
        self.MSE_u = MSE_u
        self.MSE_w = MSE_w

    def compute_gen_error(self):
        gen_error = gen_err_class(non_linearity_1=self.non_linearity_1,
                                  non_linearity_2=self.non_linearity_2, Delta_1=self.Delta_1, Delta_2=self.Delta_2)
        self.gen_error = gen_error.gen_err_explicit(
            self.q_v, self.q_w)

    def generate_non_informative_matrix(self):
        mode = 'random_weights'
        if mode == 'random_weights':
            self.coef_initialization_random = 1e-2
            self.N = 100
            x = np.random.random((self.K, self.N))
            X = self.coef_initialization_random * 1/self.N * x.dot(x.T)
        elif mode == 'random_overlap':
            X = make_spd_matrix(self.K)
            # X = np.random.random((self.K, self.K))
            X /= 100 * np.amax(X)
        else:
            raise NameError('Undefinded mode')
        return X

    def generate_informative_matrix(self):
        X = (self.initialization_coef_d-self.initialization_coef_a) * self.idty + self.initialization_coef_a * \
            np.random.uniform() * np.ones((self.K, self.K))
        return X

    ## Saddle point eq_uations ##
    def SP_q_w(self):
        if self.weights == "gaussian":
            res = self.idty - inv(self.idty + self.q_w_hat)
        elif self.weights == 'binary':
            res = self.integration_SP_prior(mode='fw')
        else:
            raise NameError('SP_q_w not defined for this prior')
        return res

    def SP_q_v(self):
        if self.weights == "gaussian":
            if self.Delta == 0:
                res = self.idty
            else:
                if self.structured_prior_mode:
                    gamma_u = 1 / self.Delta * self.q_u
                else:
                    gamma_u = self.alpha_2 / self.Delta * self.q_u
                res = self.idty - inv(self.idty + gamma_u)
        elif self.weights == 'binary':
            if self.Delta > 1e-2:
                res = self.integration_SP_prior(mode='fv')
            else:
                res = 1
        else:
            raise NameError('SP_q_v not defined for this prior')
        return res

    def SP_q_w_hat(self):
        if self.alpha_2 == 0:
            return np.zeros((self.K, self.K))
        if self.non_linearity_2 == "linear":
            if self.explicit_integral:
                gamma_v = self.alpha_1 / self.Delta * self.q_v
                res = self.alpha_2 * \
                    inv((self.Q0_w-self.q_w) + inv(gamma_v) +
                        self.Delta_2 * self.idty)
            else:
                res = self.alpha_2 * self.integration_SP_channel(mode='fout')
        elif self.non_linearity_2 == 'relu' or self.non_linearity_2 == 'sign':
            res = self.alpha_2 * self.integration_SP_channel(mode='fout')
        else:
            raise NameError('SP_q_w_hat not defined for this non linearity')
        return res

    def SP_q_u(self):
        if self.non_linearity_2 == "linear":
            if self.explicit_integral:
                gamma_v = self.alpha_1 / self.Delta * self.q_v
                inv_gamma_v = inv(gamma_v)
                X = inv_gamma_v + self.Delta_2 * \
                    self.idty + (self.Q0_w-self.q_w)
                Y = inv(gamma_v + inv(self.Delta_2 *
                                      self.idty + (self.Q0_w-self.q_w)))
                res = Y.dot(gamma_v.dot(X.dot(gamma_v.dot(Y)))) + self.q_w
            else:
                res = self.integration_SP_channel(mode='fu')
        elif self.non_linearity_2 == 'relu' or self.non_linearity_2 == 'sign':
            res = self.integration_SP_channel(mode='fu')
        else:
            raise NameError('SP_q_u not defined for this non linearity')
        return res

    ## Compute fout ##
    def compute_fout_gout(self, u, xi):
        if self.method_gout == 'explicit':
            if self.non_linearity_2 == 'relu' and self.K == 1:
                fout_gout_relu = fout_gout_dgout_relu(Delta_2=self.Delta_2, invA_B=u, A=self.alpha_1 / self.Delta * self.q_v, omega=sqrt(
                    self.q_w) * xi, V=self.Q0_w - self.q_w)
                fout, gout = fout_gout_relu.compute_fout_gout()
                #fout, gout = fout_gout_relu.compute_fout_gout_() , 1
            elif self.non_linearity_2 == 'sign':
                if self.K == 1:
                    fout_gout_sign = fout_gout_dgout_sign(Delta_2=self.Delta_2, invA_B=u, A=self.alpha_1 / self.Delta * self.q_v, omega=sqrt(
                        self.q_w) * xi, V=self.Q0_w - self.q_w)
                elif self.K == 2:
                    A = self.alpha_1 / self.Delta * self.q_v
                    inv_A = inv(A)
                    omega = sqrtm(self.q_w).dot(xi)
                    V = self.Q0_w - self.q_w
                    fout_gout_sign = fout_gout_dgout_sign_K2(
                        K=self.K, Delta_2=self.Delta_2_mat, invA_B=u, A=A, omega=omega, V=V, inv_A=inv_A)
                else:
                    raise NameError('Explicit fout undefined')
                fout, gout = fout_gout_sign.compute_fout_gout()
                #fout = fout_gout_sign.compute_fout_gout_()
                #gout = 1
            elif self.non_linearity_2 == 'linear':
                fout_gout_linear = fout_gout_dgout_linear(Delta_2=self.Delta_2, invA_B=u, A=self.alpha_1 / self.Delta * self.q_v, omega=sqrtm(
                    self.q_w).dot(xi), V=self.Q0_w - self.q_w)
                fout, gout = fout_gout_linear.compute_fout_gout()
            else:
                raise NameError('Undefined fout_gout')
        else:
            Delta_2 = self.Delta_2 * np.identity(self.K)
            invA_B = u
            A = self.alpha_1 / self.Delta * self.q_v
            omega = sqrtm(self.q_w).dot(xi)
            V = self.Q0_w - self.q_w
            inv_A_U = inv(A)
            fouf_gout_MC = fout_gout_dgout_MC(K=self.K, Delta_2=Delta_2, invA_B=invA_B, A=A, omega=omega,
                                              V=V, inv_A=inv_A_U, non_linearity_2=self.non_linearity_2, N_iter_MCquad=self.N_iter_MC_gout, NProcs=self.N_procs_MC_gout)
            fout, gout = fouf_gout_MC.compute_fout_gout()

        return fout, gout

    ## Compute fu ##
    def compute_fu_0_1(self, u, xi):
        if self.method_gout == 'explicit':
            if self.non_linearity_2 == 'relu':
                fu_0_1_relu = fu_0_1_2_relu(Delta_2=self.Delta_2, invA_B=u, A=self.alpha_1 / self.Delta * self.q_v, omega=sqrt(
                    self.q_w) * xi, V=self.Q0_w - self.q_w,)
                fu_0, fu_1 = fu_0_1_relu.compute_fu_0_1()
                #fu_0, fu_1 = fu_0_1_relu.compute_fu_0_1_(), 1
                return fu_0, fu_1
            elif self.non_linearity_2 == 'sign':
                if self.K == 1:
                    fu_0_1_sign = fu_0_1_2_sign(Delta_2=self.Delta_2, invA_B=u, A=self.alpha_1 / self.Delta * self.q_v, omega=sqrt(
                        self.q_w) * xi, V=self.Q0_w - self.q_w,)
                elif self.K == 2:
                    A = self.alpha_1 / self.Delta * self.q_v
                    inv_A = inv(A)
                    omega = sqrtm(self.q_w).dot(xi)
                    V = self.Q0_w - self.q_w
                    fu_0_1_sign = fu_0_1_2_sign_K2(
                        K=self.K, Delta_2=self.Delta_2_mat, invA_B=u, A=A, omega=omega, V=V, inv_A=inv_A)
                else:
                    raise NameError('Explicit fu undefined')
                fu_0, fu_1 = fu_0_1_sign.compute_fu_0_1()
                #fu_0, fu_1 = fu_0_1_sign.compute_fu_0_1_(), 1
                return fu_0, fu_1
            elif self.non_linearity_2 == 'linear':
                fu_0_1_linear = fu_0_1_2_linear(Delta_2=self.Delta_2, invA_B=u, A=self.alpha_1 / self.Delta * self.q_v, omega=sqrtm(
                    self.q_w).dot(xi), V=self.Q0_w - self.q_w,)
                fu_0, fu_1 = fu_0_1_linear.compute_fu_0_1()
                return fu_0, fu_1
            else:
                raise NameError('Undefined fu_0_1')
        else:
            Delta_2 = self.Delta_2 * np.identity(self.K)
            invA_B = u
            A = self.alpha_1 / self.Delta * self.q_v
            omega = sqrtm(self.q_w).dot(xi)
            V = self.Q0_w - self.q_w
            inv_A_U = inv(A)
            fu_0_1_MC = fu_0_1_2_MC(K=self.K, Delta_2=Delta_2, invA_B=invA_B, A=A, omega=omega,
                                    V=V, inv_A=inv_A_U, non_linearity_2=self.non_linearity_2, N_iter_MCquad=self.N_iter_MC_gout, NProcs=self.N_procs_MC_gout)
            fu_0, fu_1 = fu_0_1_MC.compute_fu_0_1()
            return fu_0, fu_1

    ## Compute fv (for binary) ##
    def compute_fv_0_1(self, xi):
        Delta = 5e-2
        gamma_u = self.alpha_2 / Delta * self.q_u
        gamma_u_sqrt = sqrtm(gamma_u)
        if self.weights == 'binary':
            fv_0_1_binary = fv_fw_binary(
                K=self.K, B=gamma_u_sqrt.dot(xi), A=gamma_u)
            fv_0, fv_1 = fv_0_1_binary.compute_fv_w_0_1()
            return fv_0, fv_1
        else:
            raise NameError('Undefined weights')

    ## Compute fw (for binary) ##
    def compute_fw_0_1(self, xi):
        q_w_hat = self.q_w_hat
        q_w_hat_sqrt = sqrtm(self.q_w_hat)
        if self.weights == 'binary':
            fw_0_1_binary = fv_fw_binary(
                K=self.K, B=q_w_hat_sqrt.dot(xi), A=q_w_hat)
            fw_0, fw_1 = fw_0_1_binary.compute_fv_w_0_1()
            return fw_0, fw_1
        else:
            raise NameError('Undefined weights')

    ## Compute Saddle point with quad or MC ##
    def integration_SP_channel(self, mode='fout'):
        if mode == 'fout':
            if self.K == 1:
                integrator = self.integrator_SP_q_w_hat
            elif self.K == 2:
                integrator = self.integrator_SP_q_w_hat_K2
            else:
                raise NameError('Undefined integrator')
            integrator_MC = self.integrator_SP_q_w_hat_MC
            integrator_MC_imp = self.integrator_SP_q_w_hat_MC_imp
        elif mode == 'fu':
            if self.K == 1:
                integrator = self.integrator_SP_q_u
            elif self.K == 2:
                integrator = self.integrator_SP_q_u_K2
            else:
                raise NameError('Undefined integrator')
            integrator_MC = self.integrator_SP_q_u_MC
            integrator_MC_imp = self.integrator_SP_q_u_MC_imp
        else:
            raise NameError('Undefined mode')

        if self.method_integration == 'quad':
            opt = {}
            if self.K == 1:
                if self.non_linearity_2 == 'sign':
                    borne_inf_u, borne_sup_u, borne_inf_xi, borne_sup_xi = -np.inf, np.inf, -np.inf, np.inf
                    #borne_inf_u, borne_sup_u, borne_inf_xi, borne_sup_xi = -10, 10, -10, 10
                    #borne_inf_u, borne_sup_u, borne_inf_xi, borne_sup_xi = - 40 , 40, -5, 5
                    epsabs = 1e-6
                    opt['epsabs'] = epsabs
                    opt['epsrel'] = epsabs
                    opt['limit'] = 1000
                elif self.non_linearity_2 == 'relu':
                    borne_inf_u, borne_sup_u, borne_inf_xi, borne_sup_xi = - np.inf, np.inf, -np.inf, np.inf
                    #borne_inf_u, borne_sup_u, borne_inf_xi, borne_sup_xi = -10, 10, -10, 10
                    #borne_inf_u, borne_sup_u, borne_inf_xi, borne_sup_xi = -40, 40, -5, 5
                    epsabs = 1e-6
                    opt['epsabs'] = epsabs
                    opt['epsrel'] = epsabs
                    opt['limit'] = 1000
                else : 
                    borne_inf_u = self.borne_inf
                    borne_sup_u = self.borne_sup
                    borne_inf_xi = self.borne_inf
                    borne_sup_xi = self.borne_sup
                    borne_inf_u = -5
                    borne_sup_u = 5
                    borne_inf_xi = -5
                    borne_sup_xi = 5
                    opt = {}
                    epsabs = 1e-6
                    opt['epsabs'] = epsabs
                    opt['epsrel'] = epsabs
                    opt['limit'] = 1000

                I, I_err = nquad(integrator, [
                    [borne_inf_u, borne_sup_u], [borne_inf_xi, borne_sup_xi]], opts=opt)

            elif self.K == 2:
                I = np.zeros((self.K, self.K))
                for i in range(self.K):
                    for j in range(self.K):
                        print(i, j)
                        I[i, j], I_err = nquad(integrator, [
                            [self.borne_inf, self.borne_sup], [
                                self.borne_inf, self.borne_sup],
                            [self.borne_inf, self.borne_sup], [self.borne_inf, self.borne_sup]],
                            args=(i, j), opts=opt)
            else:
                raise NameError('Integrator not defined for K>1')

        elif self.method_integration == 'MC_imp':
            self.n_ = 0
            I, I_err = mcimport(
                integrator_MC_imp, self.N_iter_MC, self.sampler_mcquad_channel, nprocs=self.N_procs_MC)
            # print('imp',mode,I)

        elif self.method_integration == 'MC_unif':
            self.n_ = 0
            borne_inf = [float(self.borne_inf) for i in range(2 * self.K)]
            borne_sup = [float(self.borne_sup) for i in range(2 * self.K)]
            I, I_err = mcquad(
                integrator_MC, self.N_iter_MC, borne_inf, borne_sup, nprocs=self.N_procs_MC)

        else:
            raise NameError('Undefined method_integration ')

        return I

    def integration_SP_prior(self, mode='fv'):
        if mode == 'fv':
            if self.method_integration == 'quad':
                integrator = self.integrator_SP_q_v
            if self.method_integration == 'MC':
                integrator_MC = self.integrator_SP_q_v_MC
        elif mode == 'fw':
            if self.method_integration == 'quad':
                integrator = self.integrator_SP_q_w
                # integrator_MC = self.integrator_SP_q_w_MC
            if self.method_integration == 'MC':
                integrator_MC = self.integrator_SP_q_w_MC
        else:
            raise NameError('Undefined mode')

        if self.K == 1:
            if self.method_integration == 'quad':
                opt = {}
                opt['epsabs'] = 1e-4
                I, I_err = nquad(integrator, [
                    [self.borne_inf, self.borne_sup]], opts=opt)
            elif self.method_integration == 'MC':
                I, I_err = mcimport(
                    integrator_MC, 1000, self.sampler_mcquad_prior, nprocs=1)
            else:
                raise NameError('Undefined method_integration ')
        else:
            raise NameError('Integrator not defined for K>1')
        return I

    ## Integrator with quad ##
    def integrator_SP_q_w_hat(self, u, xi):
        if self.K == 1:
            fout, gout = self.compute_fout_gout(u, xi)
            integrator = gaussian(xi, 0, 1) * fout * gout**2
        else:
            raise NameError('Undefined SP equation q_w_hat')
        return integrator

    def integrator_SP_q_w_hat_K2(self, u1, u2, xi1, xi2, i, j):
        u = np.array([u1, u2])
        xi = np.array([xi1, xi2])
        if self.K == 2:
            fout, gout = self.compute_fout_gout(u, xi)
            integrator = gaussian(xi, 0, 1) * fout * gout[i] * gout[j]
        else:
            raise NameError('Undefined SP equation q_w_hat')
        return integrator

    def integrator_SP_q_u(self, u, xi):
        if self.K == 1:
            fu_0, fu_1 = self.compute_fu_0_1(u, xi)
            integrator = gaussian(xi, 0, 1) * fu_0 * fu_1**2
        else:
            raise NameError('Undefined SP equation q_u')
        return integrator

    def integrator_SP_q_u_K2(self, u1, u2, xi1, xi2, i, j):
        u = np.array([u1, u2])
        xi = np.array([xi1, xi2])
        if self.K == 2:
            fu_0, fu_1 = self.compute_fu_0_1(u, xi)
            integrator = gaussian(xi, 0, 1) * fu_0 * fu_1[i] * fu_1[j]
        else:
            raise NameError('Undefined SP equation q_u')
        return integrator

    def integrator_SP_q_v(self, xi):
        if self.K == 1:
            fv_0, fv_1 = self.compute_fv_0_1(xi)
            integrator = gaussian(xi, 0, 1) * fv_0 * fv_1**2
        else:
            raise NameError('Undefined SP equation q_v')
        return integrator

    def integrator_SP_q_w(self, xi):
        if self.K == 1:
            fw_0, fw_1 = self.compute_fw_0_1(xi)
            integrator = gaussian(xi, 0, 1) * fw_0 * fw_1**2
        else:
            raise NameError('Undefined SP equation q_w')
        return integrator

    ## Integrator with MC ##
    def integrator_SP_q_w_hat_MC(self, u_xi):
        self.n_ += 1
        if (self.n_/self.N_iter_MC * 100 % 10 == 0) and self.print_gout:
            print(self.n_/self.N_iter_MC*100., '%')

        u = u_xi[0:self.K].reshape(self.K)
        xi = u_xi[self.K:].reshape(self.K)

        if self.K == 1 or self.K == 2:
            fout, gout = self.compute_fout_gout(u, xi)
            gout = np.array(gout).reshape(self.K, 1)
            measure = gaussian(xi, np.zeros((self.K)),
                               np.identity(self.K)) * fout
            integrator_ = gout.dot(gout.T)
            integrator = measure * integrator_
        else:
            raise NameError('Undefined SP equation q_w_hat')
        return integrator

    def integrator_SP_q_w_hat_MC_imp(self, xi_z_u):
        self.n_ += 1
        if (self.n_/self.N_iter_MC * 100 % 10 == 0) and self.print_gout:
            print(self.n_/self.N_iter_MC*100., '%')

        xi = xi_z_u[0:self.K].reshape(self.K)
        z = xi_z_u[self.K:2 * self.K].reshape(self.K)
        u = xi_z_u[2 * self.K:3 * self.K].reshape(self.K)

        if self.K == 1 or self.K == 2:
            fout, gout = self.compute_fout_gout(u, xi)
            gout = np.array(gout).reshape(self.K, 1)
            integrator = gout.dot(gout.T)
        else:
            raise NameError('Undefined SP equation q_w_hat')
        return integrator

    def integrator_SP_q_u_MC(self, u_xi):
        self.n_ += 1
        if (self.n_/self.N_iter_MC * 100 % 10 == 0) and self.print_gout:
            print(self.n_/self.N_iter_MC*100., '%')

        u = u_xi[0:self.K]
        xi = u_xi[self.K:]

        if self.K == 1 or self.K == 2:
            fu_0, fu_1 = self.compute_fu_0_1(u, xi)
            fu_1 = np.array(fu_1).reshape(self.K, 1)
            integrator = gaussian(xi, np.zeros((self.K)), np.identity(
                self.K)) * fu_0 * fu_1.dot(fu_1.T)
        else:
            raise NameError('Undefined SP equation q_w_hat')
        return integrator

    def integrator_SP_q_u_MC_imp(self, xi_z_u):
        self.n_ += 1
        if (self.n_/self.N_iter_MC * 100 % 10 == 0) and self.print_gout:
            print(self.n_/self.N_iter_MC*100., '%')

        xi = xi_z_u[0:self.K]
        z = xi_z_u[self.K: 2 * self.K]
        u = xi_z_u[2 * self.K: 3 * self.K]
        fu_0, fu_1 = self.compute_fu_0_1(u, xi)
        fu_1 = np.array(fu_1).reshape(self.K, 1)
        # integrator = (fu_1.dot(fu_1.T)).reshape((1, self.K**2))
        integrator = fu_1.dot(fu_1.T)
        return integrator

    def integrator_SP_q_v_MC(self, xi):
        raise NameError('Undefined integrator_SP_q_v_MC')

    def integrator_SP_q_w_MC(self, xi):
        raise NameError('Undefined integrator_SP_q_w_MC')

    def sampler_mcquad_channel(self, size):
        if self.non_linearity_2 == 'relu':
            phi = relu
        elif self.non_linearity_2 == 'sign':
            phi = sign
        elif self.non_linearity_2 == 'linear':
            phi = linear
        else:
            raise NameError('Non linearity undefined')

        # xi
        mean_xi, var_xi = np.zeros((self.K)), np.identity(self.K)
        list_xi = (multivariate_normal.rvs(
            mean_xi, var_xi, size)).reshape(self.K, size)
        # z
        list_z = []
        for i in range(size):
            if size == 1:
                mean_z = sqrtm(self.q_w).dot(list_xi)
            else:
                mean_z = sqrtm(self.q_w).dot(list_xi[:, i])
            mean_z = mean_z.reshape(self.K)
            var_z = self.Q0_w - self.q_w
            z = (multivariate_normal.rvs(mean_z, var_z, 1)).reshape(self.K)
            list_z.append(z)
        list_z = np.array(list_z).reshape(self.K, size)
        # u_tilde
        list_u = []
        gamma_v = self.alpha_1 / self.Delta * self.q_v
        inv_gamma_v = inv(gamma_v)
        var_u = (self.Delta_2_mat + inv_gamma_v)
        for i in range(size):
            if size == 1:
                mean_u = phi(list_z)
            else:
                mean_u = phi(list_z[:, i])
            mean_u = mean_u.reshape((self.K))
            u = multivariate_normal.rvs(mean_u, var_u, 1).reshape(self.K)
            list_u.append(u)
        list_u = np.array(list_u).reshape(self.K, size)
        # xi_z_u
        xi_z_u = np.array((list_xi, list_z, list_u)).T
        return xi_z_u

    def sampler_mcquad_prior(self, size):
        xi = normal(loc=0.0, scale=1.0, size=size)
        return xi

    def test_MC(self, mode):
        d = 0.4
        a = 0.1
        self.q_u = (d-a) * np.identity(self.K) + a * np.ones((self.K, self.K))
        self.q_v = (d-a) * np.identity(self.K) + a * np.ones((self.K, self.K))
        self.q_w = (d-a) * np.identity(self.K) + a * np.ones((self.K, self.K))
        self.q_w_hat = (d-a) * np.identity(self.K) + \
            a * np.ones((self.K, self.K))
        if mode == 'fout':
            integrator_MC = self.integrator_SP_q_w_hat_MC
            integrator_MC_imp = self.integrator_SP_q_w_hat_MC_imp
        else:
            integrator_MC = self.integrator_SP_q_u_MC
            integrator_MC_imp = self.integrator_SP_q_u_MC_imp

        # Unif
        borne_inf = [float(self.borne_inf) for i in range(2 * self.K)]
        borne_sup = [float(self.borne_sup) for i in range(2 * self.K)]
        I, I_err = mcquad(
            integrator_MC, self.N_iter_MC, borne_inf, borne_sup, nprocs=self.N_procs_MC)
        print('unif', mode, I)

        # Explicit
        if mode == 'fout':
            I = inv((self.Q0_w-self.q_w) + self.Delta /
                    self.alpha_1 * inv(self.q_v) + self.Delta_2 * self.idty)
        else:
            gamma_v = self.alpha_1 / self.Delta * self.q_v
            inv_gamma_v = inv(gamma_v)
            X = inv_gamma_v + self.Delta_2 * \
                self.idty + (self.Q0_w-self.q_w)
            Y = inv(gamma_v + inv(self.Delta_2 *
                                  self.idty + (self.Q0_w-self.q_w)))
            I = Y.dot(gamma_v.dot(X.dot(gamma_v.dot(Y)))) + self.q_w
        print('explicit', mode, I)

        # Imp
        I, I_err = mcimport(
            integrator_MC_imp, self.N_iter_MC, self.sampler_mcquad_channel, nprocs=self.N_procs_MC)
        print('imp', mode, I)
        if mode == 'fu':
            sys.exit("Stop")
