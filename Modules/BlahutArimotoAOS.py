import numpy as np
import pandas as pd
import warnings

# Custom format function
def custom_formatwarning(msg, *args, **kwargs):
    return f"{msg}\n"
warnings.formatwarning = custom_formatwarning

# Mutual Information in bits
"""
Calculate the mutual information between X and Y in bits.
Args:
    py: Probability distribution of Y
    px: Probability distribution of X
    pxgy: Conditional probability of X given Y
Returns:
    Mutual information I(X;Y) in bits
"""
def mutualinformation(py, px, pxgy):
    card_y = len(py)
    card_x = len(px)
    
    if pxgy.shape != (card_x, card_y):
        raise ValueError("Dimensionality of p(x|y) does not match p(x), p(y)!")
    
    MI = 0
    for i in range(card_y):
        MI += py[i] * kl_divergence_bits(pxgy[:, i], px)
    
    return MI

# Entropy in bits
"""
Calculate the entropy of a probability distribution in bits.
Args:
    p: Probability distribution
Returns:
    Entropy in bits
"""
def entropybits(p):
    return -np.sum(np.where(p > 0.000001, p * np.log2(p), 0))

# Kullback-Leibler Divergence in bits
"""
Calculate the Kullback-Leibler divergence between two probability distributions in bits.
Args:
    p_x: First probability distribution
    p0_x: Second probability distribution (reference)
    eps: Small value to avoid division by zero
Returns:
    KL divergence in bits
"""
def kl_divergence_bits(p_x, p0_x, eps=1e-15):
    if np.any((p0_x == 0) & (p_x!=0)):
        raise ValueError("Zeros in denominator before kl_divergence computation!")
    
    p0_x_safe = np.maximum(p0_x, eps)

    with np.errstate(divide='ignore', invalid='ignore'):
        kl_div = np.sum(np.where(p_x > 0.000001, p_x * (np.log2(p_x) - np.log2(p0_x_safe)), 0))

    return kl_div

# Expected Utility
"""
Calculate the expected utility given state probabilities, action-given-state probabilities, and utility matrix.
Args:
    ps: Probability distribution of states
    pags: Conditional probability of actions given states
    U_mat: Utility matrix
Returns:
    Expected utility
"""
def expectedutility(ps, pags, U_mat):
    num_s = len(ps)
    
    if pags.shape[1] != num_s or U_mat.shape != pags.shape:
        raise ValueError("Dimensionality of p(a|s), U_mat(a,s), and p(s) does not match!")
    
    EU = 0
    for i in range(num_s):
        EU += ps[i] * np.sum(pags[:, i] * U_mat[:, i])  # Sum over all states 
    
    return EU

#Variance of utility
"""
Calculate the variance of utility.
Args:
    ps: Probability distribution of states
    pags: Conditional probability of actions given states
    U_mat: Utility matrix
    EU: Expected utility
Returns:
    Variance of utility
"""
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
"""
Calculate the Rate-Distortion objective.
Args:
    EU: Expected utility
    I_ao: Mutual information between actions and observations
    gamma_ao: Complexity cost parameter
Returns:
    Rate-Distortion objective value
"""
def RDobjective(EU, I_ao, gamma_ao):
    return EU  - gamma_ao * I_ao

# Analyze BA Solution
"""
Analyze the solution of the Blahut-Arimoto algorithm.
Args:
    scenario: Scenario parameters
    ps, po, pa: Probability distributions
    pogs, pago, pags: Conditional probabilities
    U_mat: Utility matrix
    gamma_ao: Complexity cost parameter
Returns:
    Various metrics of the solution
"""
def analyze_BAsolution( ps, po, pa, pogs, pago, pags, U_mat, gamma_ao):
    I_os = mutualinformation(ps, po, pogs)
    I_ao = mutualinformation(po, pa, pago)
    I_as = mutualinformation(ps, pa, pags)
    
    Ho = entropybits(po); Ha = entropybits(pa)
    Hogs = Ho - I_os
    Hago = Ha - I_ao
    Hags = Ha - I_as
    
    EU = expectedutility(ps, pags, U_mat)
    VarU = varutility(ps, pags, U_mat, EU)
    RDobj = RDobjective(EU, I_ao, gamma_ao)

    return I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj

"""
Compute relative exponential to avoid numerical overflow.
Args:
    val: Input values
Returns:
    Relative exponential values
"""
def relative_exp(val):
    max_exp=np.max(val)
    log_val=val-max_exp
    return np.exp(log_val) # Compute exp(val - max(val)) to avoid overflow

"""
Compute Boltzmann distribution.
Args:
    p0: Initial probability distribution
    gamma_ao: Temperature parameter
    U: Utility values
Returns:
    Boltzmann distribution
"""
def boltzmanndist(p0, gamma_ao, U):
    p_boltz = p0 * relative_exp(U / gamma_ao)
    p_boltz /= np.sum(p_boltz) # Normalize to ensure valid probability distribution
    return p_boltz

# Function to compute marginals
"""
Compute marginal probabilities for actions.
Args:
    ps: Probability distribution of states
    pogs: Conditional probability of observations given states
    pago: Conditional probability of actions given observations
Returns:
    Marginal probability of actions and conditional probability of actions given states
"""
def compute_marginals_a(ps, pogs, pago):
    pags = marginalizeo(pogs, pago)
    pa = pags @ ps
    pa += np.finfo(float).eps # Add small value to avoid division by zero
    pa /= np.sum(pa) # Normalize

    return pa, pags

# Function to marginalize over o
"""
Marginalize over observations.
Args:
    pogs: Conditional probability of observations given states
    pago: Conditional probability of actions given observations
Returns:
    Conditional probability of actions given states
"""
def marginalizeo(pogs, pago):
    num_a, num_s = pago.shape[0], pogs.shape[1]
    pags = np.zeros((num_a, num_s))

    for j in range(num_s):
        pags[:, j] = pago[:, :] @ pogs[:, j]  # Matrix multiplication for marginalization

    return pags

# Function to compute action utilities
"""
Compute action utility with optional importance sampling and risk adjustment.
Args:
    psgo: Conditional probability of states given observations
    gamma_ao: Complexity cost parameter
    U_mat: Utility matrix
    gamma_var: Risk aversion parameter
    sample_util: Number of samples for utility estimation
Returns:
    Adjusted utility, expected utility matrix, variance utility matrix, and risk charge
"""
def compute_action_utility(psgo, gamma_ao, U_mat, gamma_var=0, sample_util=0):
    num_a, num_s, num_o = U_mat.shape[0], U_mat.shape[1], psgo.shape[1]
    adjusted_utility = np.zeros((num_a, num_o))
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
            risk_charge[:,o]=variance_utility_samp / 2 / gamma_ao
        else:
            utility_to_use = expected_utility
            variance_utility_samp=variance_utility
            risk_charge[:,o]=variance_utility_samp / 2 / gamma_ao

        # Adjust the utility by subtracting the variance term
        adjusted_utility[:,o] = utility_to_use - gamma_var * risk_charge[:,o]
    
    return adjusted_utility


# Function to compute p(a|o) iteration
"""
Compute p(a|o) for one iteration of the Blahut-Arimoto algorithm.
Args:
    gamma_ao: Complexity cost parameter
    pa: Probability distribution of actions
    action_utility: Action utility matrix
Returns:
    Updated conditional probability of actions given observations
"""
def compute_pago_iteration(gamma_ao, pa, action_utility):
    num_a, num_o = action_utility.shape[0], action_utility.shape[1]
    pago = np.zeros((num_a, num_o))

    for k in range(num_o):
        pago[:, k] = boltzmanndist(pa, gamma_ao, action_utility[:,k])

    return pago

# Main function for Blahut-Arimoto iterations 
"""
Main function for Blahut-Arimoto iterations.
Args:
    scenario: Scenario parameters
    epsilon_conv: Convergence threshold
    maxiter: Maximum number of iterations
    gamma_ao: Complexity cost parameter
    gamma_var: Risk aversion parameter
    sample_util: Number of samples for utility estimation
    compute_performance: Whether to compute performance metrics
    performance_per_iteration: Whether to compute performance metrics for each iteration
    performance_as_dataframe: Whether to return performance metrics as a DataFrame
    init_pago_uniformly: Whether to initialize p(a|o) uniformly
Returns:
    Final probabilities, performance metrics (if requested)
"""
def BAiterations(scenario, epsilon_conv=0.0001, maxiter=1000, gamma_ao=1/1000, gamma_var=0,sample_util=0,
                         compute_performance=False, performance_per_iteration=False,
                         performance_as_dataframe=False, init_pago_uniformly=True):
    
    U_mat=scenario["U_mat"]
    num_acts = U_mat.shape[0]

    # Initialize p(a|o,s)
    if init_pago_uniformly:
        pa_init = np.ones(num_acts)
    else:
        pa_init = np.random.rand(num_acts)

    # Normalize the initial matrix
    pa_init /= np.sum(pa_init)

    # Call the main BA iterations function
    return BAiterations_inner(pa_init, scenario, gamma_ao, 
                                      epsilon_conv, maxiter, compute_performance=compute_performance,
                                      performance_per_iteration=performance_per_iteration,
                                      performance_as_dataframe=performance_as_dataframe,gamma_var=gamma_var,
                                      sample_util=sample_util)

"""
Inner function for Blahut-Arimoto iterations.
Args:
    pa_init: Initial probability distribution of actions
    scenario: Scenario parameters
    gamma_ao: Complexity cost parameter
    epsilon_conv: Convergence threshold
    maxiter: Maximum number of iterations
    compute_performance: Whether to compute performance metrics
    performance_per_iteration: Whether to compute performance metrics for each iteration
    performance_as_dataframe: Whether to return performance metrics as a DataFrame
    gamma_var: Risk aversion parameter
    sample_util: Number of samples for utility estimation
Returns:
    Final probabilities, performance metrics (if requested)
"""
def BAiterations_inner(pa_init, scenario, gamma_ao, epsilon_conv, maxiter,
                               compute_performance=False, performance_per_iteration=False,
                               performance_as_dataframe=False,gamma_var=0,sample_util=0):
    
    pogs=scenario["pogs"]
    U_mat=scenario["U_mat"]
    ps=scenario["ps"]
    po=scenario["po"]
    psgo=scenario["psgo"]

    adjusted_utility = compute_action_utility(psgo, gamma_ao, U_mat,gamma_var=gamma_var,sample_util=sample_util)

    pago_new = compute_pago_iteration(gamma_ao, pa_init, adjusted_utility)

    # Update the marginals
    pa_new, pags = compute_marginals_a(ps, pogs, pago_new)
    
    # Preallocate performance metrics if needed
    if performance_per_iteration:
        I_os_i = np.zeros(maxiter); I_ao_i = np.zeros(maxiter); I_as_i = np.zeros(maxiter)
        Ha_i = np.zeros(maxiter); Ho_i = np.zeros(maxiter)
        Hago_i = np.zeros(maxiter); Hogs_i = np.zeros(maxiter); Hags_i = np.zeros(maxiter)
        EU_i = np.zeros(maxiter); VarU_i = np.zeros(maxiter)
        RDobj_i = np.zeros(maxiter)

    # Main iteration loop
    for iter in range(maxiter):
        pa = pa_new
        pago = pago_new

        # Compute p(a|o,s) and p(a|o)
        pago_new = compute_pago_iteration(gamma_ao, pa, adjusted_utility)
        
        # Update the marginals
        pa_new, pags = compute_marginals_a(ps, pogs, pago_new)

        # Compute entropic quantities if requested
        if performance_per_iteration: 
            (I_os_i[iter], I_ao_i[iter], I_as_i[iter], Ho_i[iter], Ha_i[iter],
             Hogs_i[iter], Hago_i[iter], Hags_i[iter], EU_i[iter],VarU_i[iter],
             RDobj_i[iter]) = analyze_BAsolution(ps, po, pa_new, pogs, pago_new, pags, U_mat, gamma_ao)

        if iter>0: #ensure goes through process at least twice
            if (np.linalg.norm(pago.flatten() - pago_new.flatten()) < epsilon_conv):
                break

    if iter == maxiter - 1:
        warnings.warn(f"Maximum iterations reached - results might be inaccurate. gamma_ao: {gamma_ao}, gamma_var: {gamma_var}")

    # Return results
    if not compute_performance:
        return pa_new, pago_new, pags
    else:
        if not performance_per_iteration:
            # Compute performance measures for the final solution
            I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj = (
                analyze_BAsolution(ps, po, pa_new, pogs, pago_new, pags,U_mat, gamma_ao))
                
        else:
            # Trim performance metrics to actual number of iterations (works for iter=0 too)
            I_os, I_ao, I_as = [arr[:max(1, iter)] for arr in (I_os_i, I_ao_i, I_as_i)]
            Ho, Ha = [arr[:max(1, iter)] for arr in (Ho_i, Ha_i)]
            Hogs, Hago, Hags = [arr[:max(1, iter)] for arr in (Hogs_i, Hago_i, Hags_i)]
            EU, VarU, RDobj = [arr[:max(1, iter)] for arr in (EU_i, VarU_i, RDobj_i)]

        # Transform to dataframe if needed
        if not performance_as_dataframe:
            return (pa_new, pogs, pago_new, pags, I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj)
        else:
            performance_df = performancemeasures2DataFrame_var(I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj)
            return pa_new, pago_new, pags, performance_df

# Convert performance measures to DataFrame
"""
Convert performance measures to a DataFrame.
Args:
    I_os, I_ao, I_as: Mutual information measures
    Ho, Ha: Entropy measures
    Hogs, Hago, Hags: Conditional entropy measures
    EU: Expected utility
    VarU: Variance of utility
    RDobj: Rate-distortion objective value
Returns:
    DataFrame containing all performance measures
"""
def performancemeasures2DataFrame_var(I_os, I_ao, I_as, Ho, Ha, Hogs, Hago, Hags, EU, VarU, RDobj):
    return pd.DataFrame({
        "I_os": [I_os], "I_ao": [I_ao], "I_as": [I_as], "H_o": [Ho], "H_a": [Ha],
        "H_ogs": [Hogs], "H_ago": [Hago], "H_ags": [Hags], "E_U": [EU], "Var_U": [VarU], 
        "Objective_value": [RDobj]
    })