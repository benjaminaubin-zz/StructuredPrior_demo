from Library.import_library import *
from Functions.generic_functions import gaussian

class fout_gout_dgout_sign(object):
    def __init__(self, K=1, Delta_2=0, invA_B=0, A=1, omega=0, V=1, inv_A=-1, Nishimori_identity=False):
        self.K = K
        self.Delta_2 = Delta_2
        self.invA_B = invA_B
        if inv_A == -1:
            self.inv_A = inv(A)
        else:
            self.inv_A = inv_A
        # Fix for initialization
        if self.inv_A < 0:
            self.inv_A = - self.inv_A
            #print('negative inv_A')
        self.omega = omega
        self.V = V
        self.inv_V = inv(V)

        self.fout = 0
        self.gout = 0
        self.dgout = 0
        self.Nishimori_identity = Nishimori_identity
        self.threshold_zero = 1e-4

    def compute_fout_gout(self):
        self.gout_()
        return self.fout, self.gout

    def compute_fout_gout_(self):
        return self.fout_gout_()

    def fout_gout_(self):
        omega = self.omega
        V = self.V
        Delta_2 = self.Delta_2
        inv_A = self.inv_A
        invA_B = self.invA_B

        fout = 1/2 * gaussian(invA_B, 1, Delta_2 + inv_A) * (1 + erf(omega/sqrt(2*V))) \
            + 1/2 * gaussian(invA_B, -1, Delta_2 + inv_A) * \
            (1 - erf(omega/sqrt(2*V)))
            
        dfout = (gaussian(invA_B, 1, Delta_2 + inv_A) - gaussian(invA_B, -1, Delta_2 + inv_A)) * \
            gaussian(omega, 0, V)
        
        res = 1/fout * dfout**2
        return res

    def compute_gout_dgout(self):
        self.dgout_()
        return self.gout, self.dgout

    def compute_fout_gout_dgout(self):
        self.dgout_()
        return self.fout, self.gout, self.dgout

    def fout_(self):
        omega = self.omega
        V = self.V
        Delta_2 = self.Delta_2
        inv_A = self.inv_A
        invA_B = self.invA_B

        fout = 1/2 * gaussian(invA_B, 1, Delta_2 + inv_A) * (1 + erf(omega/sqrt(2*V))) \
            + 1/2 * gaussian(invA_B, -1, Delta_2 + inv_A) * \
            (1 - erf(omega/sqrt(2*V)))

        if fout < self.threshold_zero:
            fout = 0

        self.fout = fout

    def gout_(self):
        omega = self.omega
        V = self.V
        Delta_2 = self.Delta_2
        inv_A = self.inv_A
        invA_B = self.invA_B

        self.fout_()
        fout = self.fout

        dfout = (gaussian(invA_B, 1, Delta_2 + inv_A) - gaussian(invA_B, -1, Delta_2 + inv_A)) * \
            gaussian(omega, 0, V)

        if fout < self.threshold_zero:
            gout = 0
        else:
            gout = dfout / fout
        self.fout = fout
        self.dfout = dfout
        self.gout = gout

    def dgout_(self):
        omega = self.omega
        V = self.V
        Delta_2 = self.Delta_2
        inv_A = self.inv_A
        invA_B = self.invA_B
        inv_V = inv(V)

        self.gout_()
        fout = self.fout
        gout = self.gout

        if self.Nishimori_identity:
            ddfout = 0
        else:
            ddfout = (gaussian(invA_B, 1, Delta_2 + inv_A) - gaussian(invA_B, -1, Delta_2 + inv_A)) * \
                gaussian(omega, 0, V) * (-omega/V)

        if fout < self.threshold_zero:
            dgout = 0
        else:
            dgout = 1/fout * ddfout - gout**2
        self.ddfout = ddfout
        self.dgout = dgout

class fu_0_1_2_sign(object):
    def __init__(self, K=1, Delta_2=0, invA_B=0, A=1, omega=0, V=1, inv_A=-1):
        self.K = K
        self.Delta_2 = Delta_2
        self.invA_B = invA_B
        if inv_A == -1:
            self.inv_A = inv(A)
        else:
            self.inv_A = inv_A
        # Fix for initialization
        if self.inv_A < 0:
            self.inv_A = - self.inv_A
            #print('negative inv_A')

        self.omega = omega
        self.V = V
        self.inv_V = inv(V)
        self.inv_Delta2_A = inv(self.Delta_2 + self.inv_A)
        self.fu_0 = 0
        self.fu_1 = 0
        self.fu_2 = 0

        self.threshold_zero = 1e-4

    def compute_fu_0_1(self):
        self.fu_1_()
        return self.fu_0, self.fu_1

    def compute_fu_0_1_(self):
        return self.fu_0_1_()

    def compute_fu_1_2(self):
        self.fu_2_()
        return self.fu_1, self.fu_2

    def compute_fu_0_1_2(self):
        self.fu_2_()
        return self.fu_0, self.fu_1, self.fu_2

    def fu_0_1_(self):
        omega = self.omega
        V = self.V
        Delta_2 = self.Delta_2
        inv_A = self.inv_A
        invA_B = self.invA_B
        inv_Delta2_A = self.inv_Delta2_A

        fu_0 = 1/2 * gaussian(invA_B, 1, Delta_2 + inv_A) * (1 + erf(omega/sqrt(2*V))) \
            + 1/2 * gaussian(invA_B, -1, Delta_2 + inv_A) * \
            (1 - erf(omega/sqrt(2*V))) 

        term1 = 1/2 * inv_Delta2_A * \
            (1 - invA_B) * gaussian(invA_B, 1, Delta_2 + inv_A) * \
            (1 + erf(omega/sqrt(2*V)))
        term2 = 1/2 * inv_Delta2_A * \
            (-1 - invA_B) * gaussian(invA_B, -1,
                                     Delta_2 + inv_A) * (1 - erf(omega/sqrt(2*V)))
        dfu_0 = term1 + term2
        res = 1 / fu_0 * (dfu_0)**2
        return res

    def fu_0_(self):
        omega = self.omega
        V = self.V
        Delta_2 = self.Delta_2
        inv_A = self.inv_A
        invA_B = self.invA_B

        fu_0 = 1/2 * gaussian(invA_B, 1, Delta_2 + inv_A) * (1 + erf(omega/sqrt(2*V))) \
            + 1/2 * gaussian(invA_B, -1, Delta_2 + inv_A) * \
            (1 - erf(omega/sqrt(2*V)))
        if np.abs(fu_0) < self.threshold_zero:
            fu_0 = 0
        self.fu_0 = fu_0

    def fu_1_(self):
        omega = self.omega
        V = self.V
        Delta_2 = self.Delta_2
        inv_A = self.inv_A
        invA_B = self.invA_B
        inv_Delta2_A = self.inv_Delta2_A

        self.fu_0_()
        fu_0 = self.fu_0

        term1 = 1/2 * inv_Delta2_A * \
            (1 - invA_B) * gaussian(invA_B, 1, Delta_2 + inv_A) * \
            (1 + erf(omega/sqrt(2*V)))
        term2 = 1/2 * inv_Delta2_A * \
            (-1 - invA_B) * gaussian(invA_B, -1,
                                     Delta_2 + inv_A) * (1 - erf(omega/sqrt(2*V)))
        dfu_0 = term1 + term2

        if np.abs(fu_0) < self.threshold_zero:
            fu_1 = 0
        else:
            fu_1 = inv_A * dfu_0 / fu_0 + invA_B
        self.fu_0 = fu_0
        self.dfu_0 = dfu_0
        self.fu_1 = fu_1

    def fu_2_(self):
        omega = self.omega
        V = self.V
        Delta_2 = self.Delta_2
        inv_A = self.inv_A
        invA_B = self.invA_B
        inv_Delta2_A = self.inv_Delta2_A

        self.fu_1_()
        fu_0 = self.fu_0
        fu_1 = self.fu_1

        term1 = - 1/2 * inv_Delta2_A * \
            gaussian(invA_B, 1, Delta_2 + inv_A) * (1 + erf(omega/sqrt(2*V)))
        term2 = 1/2 * (inv_Delta2_A * (1 - invA_B))**2 * gaussian(invA_B, 1, Delta_2 + inv_A) * \
            (1 + erf(omega/sqrt(2*V)))
        term3 = - 1/2 * inv_Delta2_A * \
            gaussian(invA_B, -1, Delta_2 + inv_A) * (1 - erf(omega/sqrt(2*V)))
        term4 = 1/2 * (inv_Delta2_A * (-1 - invA_B))**2 * gaussian(invA_B, -1, Delta_2 + inv_A) * \
            (1 - erf(omega/sqrt(2*V)))
        ddfu_0 = term1 + term2 + term3 + term4

        if fu_0 < self.threshold_zero:
            fu_2 = 0
        else:
            fu_2 = inv_A**2 * 1/fu_0 * ddfu_0 + inv_A - \
                invA_B**2 + 2 * invA_B * fu_1 - fu_1**2

        self.fu_2 = fu_2





class fout_gout_dgout_sign_K2(object):
    def __init__(self, K=2, Delta_2=0, A=1, invA_B=0, omega=0, V=1, inv_A=-1):
        self.K = K
        self.Delta_2 = Delta_2 * np.identity(self.K)
        self.invA_B = invA_B
        self.inv_A = inv_A

        self.omega = omega
        self.V = V
        self.inv_V = inv(V)
        self.det_V = det(V)

        self.mu = self.invA_B
        self.Sigma = self.Delta_2 + self.inv_A
        self.inv_Sigma = inv(self.Sigma)
        self.det_Sigma = det(self.Sigma)

        self.V_tilde = np.array([self.V[0, 0], self.V[1, 1]])
        self.V_tilde_inv = self.det_V / self.V_tilde

        self.borne_inf = -3
        self.borne_sup = 3
        self.fout = 0
        self.gout = np.zeros((self.K))
        self.dgout = np.zeros((self.K, self.K))
        self.threshold_zero = 1e-4

    def configuration_binary(self, n):
        X = np.array([int(x) for x in list('{0:0b}'.format(n).zfill(self.K))])
        X = 2 * X - 1
        return X

    def select_indices(self, i):
        j = int(np.abs(1-i))
        return (i, j)

    def select_bornes(self, s):
        if s == -1:
            return (self.borne_inf, 0)
        elif s == 1:
            return (0, self.borne_sup)
        else:
            raise NameError('Wrong sign')

    def fout_sign_K2(self):
        fout = 0
        for i in range(self.K**2):
            phi_z = self.configuration_binary(i)
            norm = gaussian(self.mu, phi_z, self.Sigma)
            s = phi_z[1]
            borne_inf, borne_sup = self.select_bornes(phi_z[0])
            I = quad(self.integrand_fout, borne_inf, borne_sup,
                     args=(s))[0]
            I *= norm
            fout += I
        self.fout = fout
        return self.fout

    def dfout_sign_K2(self):
        dfout = np.zeros((self.K))
        for i in range(self.K**2):
            phi_z = self.configuration_binary(i)
            norm = gaussian(self.mu, phi_z, self.Sigma)
            s = phi_z[1]
            borne_inf, borne_sup = self.select_bornes(phi_z[0])
            for j in range(self.K):
                dfout[j] += norm * quad(self.integrand_dfout, borne_inf, borne_sup,
                                        args=(s, j))[0]
        return dfout

    def ddfout_sign_K2(self):
        ddfout = np.zeros((self.K, self.K))
        for i in range(self.K**2):
            phi_z = self.configuration_binary(i)
            norm = gaussian(self.mu, phi_z, self.Sigma)
            s = phi_z[1]
            borne_inf, borne_sup = self.select_bornes(phi_z[0])
            for j in range(self.K):
                for k in range(self.K):
                    ddfout[j, k] += norm * quad(self.integrand_ddfout, borne_inf, borne_sup,
                                                args=(s, j, k))[0]
        return ddfout

    def integrand_fout(self, z, s):
        A, B, C = self.omega[0], self.V_tilde_inv[1], self.V_tilde[1]
        i = 1
        omega_tilde = self.omega_tilde(z, i, self.omega, self.inv_V)
        res = 1/2 * gaussian(z, A, B) * (1 + s *
                                         erf(omega_tilde / sqrt(2*C)))
        return res

    def integrand_dfout(self, z, s, j):
        A, B, C = self.omega[0], self.V_tilde_inv[1], self.V_tilde[1]
        i = 1
        omega_tilde = self.omega_tilde(z, i, self.omega, self.inv_V)
        term1 = 1/2 * (1 + s * erf(omega_tilde / sqrt(2*C))) * 1 / \
            self.V_tilde_inv[1] * (z - self.omega[0]) if j == 0 else 0
        tmp = self.inv_V[0, 1] / self.inv_V[1, 1] if j == 0 else 1
        term2 = s * gaussian(0, omega_tilde, self.V_tilde[1]) * tmp
        res = gaussian(z, A, B) * (term1 + term2)
        return res

    def integrand_ddfout(self, z, s, j, k):
        A, B, C = self.omega[0], self.V_tilde_inv[1], self.V_tilde[1]
        i = 1
        omega_tilde = self.omega_tilde(z, i, self.omega, self.inv_V)
        term0 = 1 / self.V_tilde_inv[1] * (z - self.omega[0]) if k == 0 else 0
        term1 = 1/2 * (1 + s * erf(omega_tilde / sqrt(2*C))) * 1 / \
            self.V_tilde_inv[1] * (z - self.omega[0]) if j == 0 else 0
        tmp = self.inv_V[0, 1] / self.inv_V[1, 1] if j == 0 else 1
        term2 = s * gaussian(0, omega_tilde, self.V_tilde[1]) * tmp
        term3 = - 1/2 * (1 + s * erf(omega_tilde / sqrt(2*C))) * 1 / \
            self.V_tilde_inv[1] if (j == 0 and k == 0) else 0
        term4 = 1 / self.V_tilde_inv[1] * (z - self.omega[0]) if j == 0 else 0
        tmp2 = self.inv_V[0, 1] / self.inv_V[1, 1] if k == 0 else 1
        term5 = s * gaussian(0, omega_tilde, self.V_tilde[1]) * tmp2
        term6 = - s * \
            gaussian(0, omega_tilde, self.V_tilde[1]) * \
            tmp * 1 / self.V_tilde[1] * omega_tilde * tmp2
        res = gaussian(z, A, B) * (term0 * (term1 + term2) +
                                   term3 + term4 * term5 + term6)
        return res

    def gout_sign_K2(self):
        self.fout_sign_K2()
        if np.abs(self.fout) < self.threshold_zero:
            return self.gout
        else:
            dfout = self.dfout_sign_K2()
            self.gout = dfout/self.fout
        return self.gout

    def dgout_sign_K2(self):
        self.fout_sign_K2()
        if np.abs(self.fout) < self.threshold_zero:
            return self.dgout
        else:
            gout = self.gout_sign_K2()
            ddfout = self.ddfout_sign_K2()
            self.dgout = ddfout/self.fout - \
                gout.reshape(self.K, 1).dot(gout.reshape(1, self.K))
        return self.gout

    def omega_tilde(self, z, i, omega, inv_V):
        """
        returns omega[i] + (omega[j] - z) * inv_V[i,j]/inv_V[i,i]
        """
        (i, j) = self.select_indices(i)
        res = omega[i] - inv_V[i, j] / inv_V[i, i] * (z - omega[j])
        return res

    def compute_fout_gout(self):
        self.gout_sign_K2()
        return self.fout, self.gout

    def compute_gout_dgout(self):
        self.dgout_sign_K2()
        return self.gout, self.dgout

    def compute_fout_gout_dgout(self):
        self.dgout_sign_K2()
        return self.fout, self.gout, self.dgout


class fu_0_1_2_sign_K2(object):
    def __init__(self, K=2, Delta_2=0, A=1, invA_B=0, omega=0, V=1, inv_A=-1):
        self.K = K
        self.Delta_2 = Delta_2 * np.identity(self.K)
        self.invA_B = invA_B
        self.inv_A = inv_A

        self.omega = omega
        self.V = V
        self.inv_V = inv(V)
        self.det_V = det(V)

        self.mu = self.invA_B
        self.Sigma = self.Delta_2 + self.inv_A
        self.inv_Sigma = inv(self.Sigma)
        self.det_Sigma = det(self.Sigma)

        self.V_tilde = np.array([self.V[0, 0], self.V[1, 1]])
        self.V_tilde_inv = self.det_V / self.V_tilde

        self.borne_inf = -3
        self.borne_sup = 3
        self.fu_0 = 0
        self.fu_1 = np.zeros((self.K))
        self.fu_2 = np.zeros((self.K, self.K))
        self.threshold_zero = 1e-4

    def configuration_binary(self, n):
        X = np.array([int(x) for x in list('{0:0b}'.format(n).zfill(self.K))])
        X = 2 * X - 1
        return X

    def select_indices(self, i):
        j = int(np.abs(1-i))
        return (i, j)

    def select_bornes(self, s):
        if s == -1:
            return (self.borne_inf, 0)
        elif s == 1:
            return (0, self.borne_sup)
        else:
            raise NameError('Wrong sign')

    def fu_0_sign_K2(self):
        fu_0 = 0
        for i in range(self.K**2):
            phi_z = self.configuration_binary(i)
            norm = gaussian(self.mu, phi_z, self.Sigma)
            s = phi_z[1]
            borne_inf, borne_sup = self.select_bornes(phi_z[0])
            I = quad(self.integrand_fout, borne_inf, borne_sup,
                     args=(s))[0]
            I *= norm
            fu_0 += I
        self.fu_0 = fu_0
        return self.fu_0

    def dfu_0_sign_K2(self):
        dfu_0 = np.zeros((self.K))
        for i in range(self.K**2):
            phi_z = self.configuration_binary(i)
            norm = gaussian(self.mu, phi_z, self.Sigma)
            s = phi_z[1]
            borne_inf, borne_sup = self.select_bornes(phi_z[0])
            I = norm * quad(self.integrand_fout, borne_inf, borne_sup,
                            args=(s))[0]
            dfu_0 += self.inv_Sigma.dot(phi_z-self.mu) * I
        return dfu_0

    def ddfu_0_sign_K2(self):
        ddfu_0 = np.zeros((self.K, self.K))
        for i in range(self.K**2):
            phi_z = self.configuration_binary(i)
            norm = gaussian(self.mu, phi_z, self.Sigma)
            s = phi_z[1]
            borne_inf, borne_sup = self.select_bornes(phi_z[0])
            I = norm * quad(self.integrand_fout, borne_inf, borne_sup,
                            args=(s))[0]
            X = (self.inv_Sigma).dot(phi_z-self.mu).reshape(self.K, 1)
            ddfu_0 += (- self.inv_Sigma + X.dot(X.T)) * I
        return ddfu_0

    def fu_1_sign_K2(self):
        fu_0 = self.fu_0_sign_K2()
        if fu_0 < self.threshold_zero : 
            return self.fu_1
        else :
            dfu_0 = self.dfu_0_sign_K2()
            fu_1 = 1 / fu_0 * self.inv_A.dot(dfu_0) + self.mu
            self.fu_1 = fu_1
            return fu_1

    def fu_2_sign_K2(self):
        fu_0 = self.fu_0_sign_K2()
        if fu_0 < self.threshold_zero:
            return self.fu_2
        else:
            fu_1 = self.fu_1_sign_K2().reshape(self.K, 1)
            ddfu_0 = self.ddfu_0_sign_K2()
            mu = self.mu.reshape(self.K, 1)
            fu_2 = 1 / fu_0 * self.inv_A.dot(ddfu_0).dot(self.inv_A) + self.inv_A - \
                mu.dot(mu.T) + fu_1.dot(mu.T) + mu.dot(fu_1.T) - fu_1.dot(fu_1.T)
            self.fu_2 = fu_2
            return fu_2

    def ddfout_sign_K2(self):
        ddfout = np.zeros((self.K, self.K))
        for i in range(self.K**2):
            phi_z = self.configuration_binary(i)
            norm = gaussian(self.mu, phi_z, self.Sigma)
            s = phi_z[1]
            borne_inf, borne_sup = self.select_bornes(phi_z[0])
            for j in range(self.K):
                for k in range(self.K):
                    ddfout[j, k] += norm * quad(self.integrand_ddfout, borne_inf, borne_sup,
                                                args=(s, j, k))[0]
        return ddfout

    def integrand_fout(self, z, s):
        A, B, C = self.omega[0], self.V_tilde_inv[1], self.V_tilde[1]
        i = 1
        omega_tilde = self.omega_tilde(z, i, self.omega, self.inv_V)
        res = 1/2 * gaussian(z, A, B) * (1 + s *
                                         erf(omega_tilde / sqrt(2*C)))
        return res

    def integrand_dfout(self, z, s, j):
        A, B, C = self.omega[0], self.V_tilde_inv[1], self.V_tilde[1]
        i = 1
        omega_tilde = self.omega_tilde(z, i, self.omega, self.inv_V)
        term1 = 1/2 * (1 + s * erf(omega_tilde / sqrt(2*C))) * 1 / \
            self.V_tilde_inv[1] * (z - self.omega[0]) if j == 0 else 0
        tmp = self.inv_V[0, 1] / self.inv_V[1, 1] if j == 0 else 1
        term2 = s * gaussian(0, omega_tilde, self.V_tilde[1]) * tmp
        res = gaussian(z, A, B) * (term1 + term2)
        return res

    def integrand_ddfout(self, z, s, j, k):
        A, B, C = self.omega[0], self.V_tilde_inv[1], self.V_tilde[1]
        i = 1
        omega_tilde = self.omega_tilde(z, i, self.omega, self.inv_V)
        term0 = 1 / self.V_tilde_inv[1] * (z - self.omega[0]) if k == 0 else 0
        term1 = 1/2 * (1 + s * erf(omega_tilde / sqrt(2*C))) * 1 / \
            self.V_tilde_inv[1] * (z - self.omega[0]) if j == 0 else 0
        tmp = self.inv_V[0, 1] / self.inv_V[1, 1] if j == 0 else 1
        term2 = s * gaussian(0, omega_tilde, self.V_tilde[1]) * tmp
        term3 = - 1/2 * (1 + s * erf(omega_tilde / sqrt(2*C))) * 1 / \
            self.V_tilde_inv[1] if (j == 0 and k == 0) else 0
        term4 = 1 / self.V_tilde_inv[1] * (z - self.omega[0]) if j == 0 else 0
        tmp2 = self.inv_V[0, 1] / self.inv_V[1, 1] if k == 0 else 1
        term5 = s * gaussian(0, omega_tilde, self.V_tilde[1]) * tmp2
        term6 = - s * \
            gaussian(0, omega_tilde, self.V_tilde[1]) * \
            tmp * 1 / self.V_tilde[1] * omega_tilde * tmp2
        res = gaussian(z, A, B) * (term0 * (term1 + term2) +
                                   term3 + term4 * term5 + term6)
        return res

    def gout_sign_K2(self):
        self.fout_sign_K2()
        if np.abs(self.fout) < self.threshold_zero:
            return self.gout
        else:
            dfout = self.dfout_sign_K2()
            self.gout = dfout/self.fout
        return self.gout

    def dgout_sign_K2(self):
        self.fout_sign_K2()
        if np.abs(self.fout) < self.threshold_zero:
            return self.dgout
        else:
            gout = self.gout_sign_K2()
            ddfout = self.ddfout_sign_K2()
            self.dgout = ddfout/self.fout - \
                gout.reshape(self.K, 1).dot(gout.reshape(1, self.K))
        return self.gout

    def omega_tilde(self, z, i, omega, inv_V):
        """
        returns omega[i] + (omega[j] - z) * inv_V[i,j]/inv_V[i,i]
        """
        (i, j) = self.select_indices(i)
        res = omega[i] - inv_V[i, j] / inv_V[i, i] * (z - omega[j])
        return res

    def compute_fu_0_1(self):
        self.fu_1_sign_K2()
        return self.fu_0, self.fu_1

    def compute_fu_1_2(self):
        self.fu_2_sign_K2()
        return self.fu_1, self.fu_2

    def compute_fu_0_1_2(self):
        self.fu_2_sign_K2()
        return self.fu_0, self.fu_1, self.fu_2
