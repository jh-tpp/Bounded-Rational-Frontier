import numpy as np
from Modules.InformationTheoryFunctions import entropybits, mutualinformation, kl_divergence_bits
import pandas as pd
import warnings
# Custom format function
def custom_formatwarning(msg, *args, **kwargs):
    return f"{msg}\n"
warnings.formatwarning = custom_formatwarning

# Expected Utility
def expectedutility(ps, pags, U_mat):
    num_s = len(ps)
    num_a = pags.shape[0]
    
    if pags.shape[1] != num_s or U_mat.shape != pags.shape:
        raise ValueError("Dimensionality of p(a|s), U_mat(a,s), and p(s) does not match!")
    
    EU = 0
    for i in range(num_s):
        EU += ps[i] * np.sum(pags[:, i] * U_mat[:, i])
    
    return EU

def expectedaction(ps, pags,a_values):
    num_s = len(ps)
    
    Ea = 0
    for i in range(num_s):
        Ea += ps[i] * np.sum(pags[:, i] * a_values)
    
    return Ea

#variance of utility
def varutility(ps, pags, U_mat, EU):
    num_s = len(ps)
    num_a = pags.shape[0]

    VarU = 0
    for i in range(num_s):
        for j in range(num_a):
            # Calculate the squared difference from the expected utility
            VarU += ps[i] * pags[j, i] * (U_mat[j, i] - EU) ** 2
    
    return VarU

#Rate-Distortion Objective
def RDobjective(EU, I_ao, beta_ao):
    return EU  - (1/beta_ao) * I_ao

# Analyze BA Solution
def analyze_BAsolution(scenario, ps, po, pa, pogs, pago, pags, U_mat, beta_ao):
    I_os = mutualinformation(ps, po, pogs)
    I_ao = mutualinformation(po, pa, pago)
    I_as = mutualinformation(ps, pa, pags)
    
    Ho = entropybits(po)
    Ha = entropybits(pa)
    Hogs = Ho - I_os
    Hago = Ha - I_ao
    Hags = Ha - I_as
    
    EU = expectedutility(ps, pags, U_mat)
    VarU = varutility(ps, pags, U_mat, EU)
    RDobj = RDobjective(EU, I_ao, beta_ao)
    Ea= expectedaction(ps, pags, scenario["a_values"])

    return I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj, Ea

def setuputilityarrays(a, w, utility):
    cardinality_a = len(a)
    cardinality_w = len(w)

    # Pre-compute utilities, find maxima
    U_mat = np.zeros((cardinality_a, cardinality_w))
    
    for i in range(cardinality_w):
        for j in range(cardinality_a):
            U_mat[j, i] = utility(a[j], w[i])
    
    return U_mat


def relative_exp(val):
    max_exp=np.max(val)
    log_val=val-max_exp
    return np.exp(log_val)


def boltzmanndist(p0, beta, U):
    p_boltz = p0 * relative_exp(beta * U)
    p_boltz /= np.sum(p_boltz)
    return p_boltz

# Function to compute marginals
def compute_marginals_a(ps, pogs, pago):
    pags = marginalizeo(pogs, pago)
    pa = pags @ ps
    pa += np.finfo(float).eps
    pa /= np.sum(pa)

    return pa, pags

# Function to marginalize over o
def marginalizeo(pogs, pago):
    num_a, num_s = pago.shape[0], pogs.shape[1]
    pags = np.zeros((num_a, num_s))

    for j in range(num_s):
        pags[:, j] = pago[:, :] @ pogs[:, j]

    return pags

# Function to compute action utilities
def compute_action_utility(psgo, beta_ao, U_mat, gamma=0, sample_util=0, use_IS=False, lambda_IS=0.1):
    num_a, num_s, num_o = U_mat.shape[0], U_mat.shape[1], psgo.shape[1]
    adjusted_utility = np.zeros((num_a, num_o))
    E_U_mat=np.zeros((num_a, num_o))
    var_U_mat=np.zeros((num_a, num_o))
    risk_charge=np.zeros((num_a, num_o))

    for o in range(num_o):

        # Compute expected utility
        expected_utility = U_mat @ psgo[:,o]

        # Compute squared utility for variance calculation
        squared_U_mat = U_mat**2

        # Compute expected squared utility
        expected_squared_utility = squared_U_mat @ psgo[:,o]

        # Compute variance of utility
        variance_utility = expected_squared_utility - expected_utility**2

        if sample_util>0:
            variance_utility_samp=variance_utility / np.sqrt(sample_util)
            sampled_indices = np.random.choice(num_s, size=sample_util, p=psgo[:,o])
            # Compute the mean across the sampled columns (axis=1 means averaging over the columns for each row)
            utility_to_use = np.mean(U_mat[:, sampled_indices], axis=1)

            # Importance Sampling Option
            if use_IS:
                dU=0.000
                noisy_psgo = psgo[:, o] + 0 * np.random.randn(*psgo[:, o].shape)
                noisy_psgo = np.clip(noisy_psgo, 0, None)  # Ensure probabilities are non-negative
                noisy_psgo /= np.sum(noisy_psgo)  # Normalize to make it a valid probability distribution
                adjusted_psgo = noisy_psgo[:, np.newaxis] * np.abs(U_mat.T+dU)  # Utility-weighted adjustment
                adjusted_psgo = adjusted_psgo / np.sum(adjusted_psgo, axis=0, keepdims=True)  # Normalize per action

                risk_charge[:,o]=np.array([
                    kl_divergence_bits( adjusted_psgo[:, a], noisy_psgo)
                    for a in range(num_a)
                ])
            else:
                risk_charge[:,o]=variance_utility_samp / 2 * beta_ao
        else:
            utility_to_use = expected_utility
            variance_utility_samp=variance_utility
            risk_charge[:,o]=variance_utility_samp / 2 * beta_ao

        # Adjust the utility by subtracting the variance term
        adjusted_utility[:,o] = utility_to_use - gamma * risk_charge[:,o]
        E_U_mat[:,o]=expected_utility
        var_U_mat[:,o]=variance_utility_samp
    
    return adjusted_utility, E_U_mat, var_U_mat, risk_charge


# Function to compute p(a|o) iteration
def compute_pago_iteration(beta_ao, pa, action_utility):
    num_a, num_o = action_utility.shape[0], action_utility.shape[1]
    pago = np.zeros((num_a, num_o))

    for k in range(num_o):
        pago[:, k] = boltzmanndist(pa, beta_ao, action_utility[:,k])

    return pago

# Main function for Blahut-Arimoto iterations 

def BAiterations(scenario, epsilon_conv=0.0001, maxiter=1000, beta_ao=1000, gamma=0,sample_util=0,use_IS=False,
                         compute_performance=False, performance_per_iteration=False,
                         performance_as_dataframe=False, init_pago_uniformly=True):
    
    cardinality_obs=len(scenario["o_values"])
    pogs=scenario["pogs"]
    U_mat=scenario["U_mat"]
    ps=scenario["ps"]

    num_acts = U_mat.shape[0]
    num_worldstates = U_mat.shape[1]

    # Initialize p(a|o,s)
    if init_pago_uniformly:
        pa_init = np.ones(num_acts)
    else:
        pa_init = np.random.rand(num_acts)

    # Normalize the initial matrix
    pa_init /= np.sum(pa_init)

    # Call the main BA iterations function
    return BAiterations_inner(pa_init, scenario, beta_ao, 
                                      epsilon_conv, maxiter, compute_performance=compute_performance,
                                      performance_per_iteration=performance_per_iteration,
                                      performance_as_dataframe=performance_as_dataframe,gamma=gamma,
                                      sample_util=sample_util,use_IS=use_IS)


def BAiterations_inner(pa_init, scenario, beta_ao, epsilon_conv, maxiter,
                               compute_performance=False, performance_per_iteration=False,
                               performance_as_dataframe=False,gamma=0,sample_util=0,use_IS=False):
    
    pogs=scenario["pogs"]
    U_mat=scenario["U_mat"]
    ps=scenario["ps"]
    po=scenario["po"]
    psgo=scenario["psgo"]

    adjusted_utility, E_U_mat, var_U_mat, risk_charge =compute_action_utility(psgo, beta_ao, U_mat,gamma=gamma,sample_util=sample_util, use_IS=use_IS)


    # print(f"adjusted_utility: {adjusted_utility}")
    # print(f"risk_charge: {risk_charge}")
    # print(f"E_U_mat: {np.round(E_U_mat,decimals=2)}")

    pago_new = compute_pago_iteration(beta_ao, pa_init, adjusted_utility)
    # Update the marginals
    pa_new, pags = compute_marginals_a(ps, pogs, pago_new)

    # print(f"beta_ao: {beta_ao}")
    # print(f"pa_init: {pa_init}")
    # print(f"pogs: {pogs}")
    # print(f"ps: {ps}")
    # print(f"pago_new: {pago_new}")
    # print(f"pa_new: {pa_new}")
    
    # Preallocate performance metrics if needed
    if performance_per_iteration:
        I_os_i = np.zeros(maxiter)
        I_ao_i = np.zeros(maxiter)
        I_as_i = np.zeros(maxiter)
        Ha_i = np.zeros(maxiter)
        Ho_i = np.zeros(maxiter)
        Hago_i = np.zeros(maxiter)
        Hogs_i = np.zeros(maxiter)
        Hags_i = np.zeros(maxiter)
        EU_i = np.zeros(maxiter)
        VarU_i = np.zeros(maxiter)
        RDobj_i = np.zeros(maxiter)
        Ea_i = np.zeros(maxiter)

    # Main iteration loop
    for iter in range(maxiter):
        pa = pa_new
        pago = pago_new
        # Compute p(a|o,s) and p(a|o)
        pago_new = compute_pago_iteration(beta_ao, pa, adjusted_utility)
        # Update the marginals
        pa_new, pags = compute_marginals_a(ps, pogs, pago_new)
        # print(f"iter: {iter}")
        # print(f"pago_new: {pago_new}")
        # print(f"pa_new: {pa_new}")

        # Compute entropic quantities if requested
        if performance_per_iteration: 
            (I_os_i[iter], I_ao_i[iter], I_as_i[iter], Ho_i[iter], Ha_i[iter],
             Hogs_i[iter], Hago_i[iter], Hags_i[iter], EU_i[iter],VarU_i[iter],
             RDobj_i[iter],Ea_i[iter]) = analyze_BAsolution(scenario,ps, po, pa_new, pogs, pago_new, pags, U_mat, beta_ao)

        if iter>0: #ensure goes through process at least twice
            if (np.linalg.norm(pago.flatten() - pago_new.flatten()) < epsilon_conv):
                break

    if iter == maxiter - 1:
        warnings.warn(f"Maximum iterations reached - results might be inaccurate. beta_ao: {beta_ao}, gamma: {gamma}")

    # Return results
    if not compute_performance:
        return pa_new, pago_new, pags
    else:
        if not performance_per_iteration:
            # Compute performance measures for the final solution
            I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj, Ea = (
                analyze_BAsolution(scenario,ps, po, pa_new, pogs, pago_new, pags,U_mat, beta_ao))
                
        else:
            if iter>0:
                I_os = I_os_i[:iter]
                I_ao = I_ao_i[:iter]
                I_as = I_as_i[:iter]
                Ho = Ho_i[:iter]
                Ha = Ha_i[:iter]
                Hogs = Hogs_i[:iter]
                Hago = Hago_i[:iter]
                Hags = Hags_i[:iter]
                EU = EU_i[:iter]
                VarU = VarU_i[:iter]
                RDobj = RDobj_i[:iter]
                Ea = Ea_i[:iter]
            else:
                I_os = I_os_i[0]
                I_ao = I_ao_i[0]
                I_as = I_as_i[0]
                Ho = Ho_i[0]
                Ha = Ha_i[0]
                Hogs = Hogs_i[0]
                Hago = Hago_i[0]
                Hags = Hags_i[0]
                EU = EU_i[0]
                VarU = VarU_i[0]
                RDobj = RDobj_i[0]
                Ea = Ea_i[0]

        # Transform to dataframe if needed
        if not performance_as_dataframe:
            return (pa_new, pogs, pago_new, pags, I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj, Ea)
        else:
            performance_df = performancemeasures2DataFrame_var(I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj, Ea)
            return pa_new, pago_new, pags, performance_df, risk_charge, E_U_mat

# Convert performance measures to DataFrame
def performancemeasures2DataFrame_var(I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj, Ea):
    return pd.DataFrame({
        "I_os": [I_os], "I_ao": [I_ao], "I_as": [I_as], "H_o": [Ho], "H_a": [Ha],
        "H_ogs": [Hogs], "H_ago": [Hago], "H_ags": [Hags], "E_U": [EU], "Var_U": [VarU], 
        "Objective_value": [RDobj], "Ea": [Ea]
    })