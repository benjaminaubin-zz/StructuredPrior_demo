from Class.AMP_StructuredPrior import AMP_averaged
from Class.SE_structuredPrior import StateEvolution
from Class.Spectral_structuredPrior import Spectral_averaged


def run_AMP(N=3000, alpha=1, beta=1, non_linearity='linear', Delta=1, seed=False, save=False, N_average=1, initialization_mode='planted', model='UU', verbose='False'):
    print(f'#### START AMP ####')
    AMP_avg = AMP_averaged(K=1,  N=N, alpha_1=beta, alpha_2=alpha, non_linearity_1='linear', non_linearity_2=non_linearity,
                           weights='gaussian', Delta_1=Delta, Delta_2=0, seed=seed, save=save, N_average=N_average,
                           method_gout='explicit', initialization_mode=initialization_mode, model=model, verbose=verbose)
    AMP_avg.main()
    print(f'#### END AMP #### \n')
    return AMP_avg.q_v, AMP_avg.MSE_v


def run_SE(alpha=1, beta=1, non_linearity='linear', Delta=1, save=False, model='UU', verbose='False'):
    print(f'#### START SE ####')
    SE = StateEvolution(K=1, alpha_1=beta, alpha_2=alpha,
                        non_linearity_1='linear', non_linearity_2=non_linearity, weights='gaussian',
                        Delta_1=Delta, Delta_2=0, method_integration='quad', method_gout='explicit',
                        initialization_mode='informative', model=model, verbose=verbose)
    SE.main()
    print(f'#### END SE #### \n')
    return SE.q_v, SE.MSE_v


def run_Spectral(N=2000, alpha=1, beta=1, non_linearity='linear', Delta=1, seed=False, save=False, N_average=1, model='UU', verbose='False'):
    print(f'#### START Spectral ####')
    Spectral = Spectral_averaged(K=1, N=N, alpha_1=beta, alpha_2=alpha, non_linearity_1='linear', non_linearity_2=non_linearity,
                                 weights='gaussian', Delta_1=Delta, Delta_2=0, seed=seed, save=save, model=model, N_average=N_average, verbose=verbose)
    Spectral.main()
    print(f'#### END Spectral #### \n')
    return Spectral.q_v_PCA, Spectral.q_v_lAMP, Spectral.MSE_PCA, Spectral.MSE_lAMP


def run_all(N=3000, alpha=1, beta=1, non_linearity='linear', Delta=1, N_average=1, seed=False, save=False, initialization_mode='planted', model='UU', verbose='False'):
    _, MSEv_AMP = run_AMP(N, alpha, beta, non_linearity, Delta,
                          seed, save, N_average, initialization_mode, model, verbose)
    _, MSEv_SE = run_SE(alpha, beta, non_linearity,
                        Delta, save, model, verbose)
    _, _, MSEv_PCA, MSEv_lAMP = run_Spectral(
        N, alpha, beta, non_linearity, Delta, seed, save, N_average, model, verbose)
    return MSEv_AMP, MSEv_SE, MSEv_PCA, MSEv_lAMP
