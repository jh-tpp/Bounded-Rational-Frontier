import numpy as np
import pandas as pd
import itertools
from Modules.BlahutArimotoAOS import BAiterations


def setup_scenarios(da=1,v=10):
    a_values, a_strings, s_values, s_strings, U_mat = setup_U_mat(a_step=da, vu=v)  # Recalculate U_mat for each v

    scenarios = [
        {"id": 0, "name": "Perfect inference", "a_step": da, "p1a": 0.009, "p1b": 0.99, "p2a": 0.009, "p2b": 0.99, "cor12": 0, "a_values": a_values, "a_strings": a_strings, "s_values": s_values, "s_strings": s_strings, "U_mat": U_mat},
        {"id": 1, "name": "Good, independent observations", "a_step": da, "p1a": 0.3, "p1b": 0.65, "p2a": 0.3, "p2b": 0.65, "cor12": 0, "a_values": a_values, "a_strings": a_strings, "s_values": s_values, "s_strings": s_strings, "U_mat": U_mat},
        {"id": 2, "name": "Correlated observations", "a_step": da, "p1a": 0.3, "p1b": 0.65, "p2a": 0.3, "p2b": 0.65, "cor12": 0.5, "a_values": a_values, "a_strings": a_strings, "s_values": s_values, "s_strings": s_strings, "U_mat": U_mat},
        {"id": 3, "name": "Only one good observation", "a_step": da, "p1a": 0.3, "p1b": 0.65, "p2a": 0.3333, "p2b": 0.3334, "cor12": 0, "a_values": a_values, "a_strings": a_strings, "s_values": s_values, "s_strings": s_strings, "U_mat": U_mat},
        {"id": 4, "name": "Relatively clueless", "a_step": da, "p1a": 0.33, "p1b": 0.4, "p2a": 0.3333, "p2b": 0.3334, "cor12": 0, "a_values": a_values, "a_strings": a_strings, "s_values": s_values, "s_strings": s_strings, "U_mat": U_mat},
    ]

    return scenarios

def setup_U_mat(a_step=0.5,vu=1):

    # Generate a_values from 0 to 1.0 in steps of a_step
    a_values = np.arange(0, 1.0 + a_step, a_step)
    a_strings = [f"a={round(a, 1)}" for a in a_values]

    # Define the possible values for v
    v_values = [0, vu]
    s_combinations = list(itertools.product(v_values))
    s_values = v_values
    s_strings = [f"s={s[0]}" for s in s_combinations]

    # Initialize U_mat with dimensions (len(a_values) x len(s_combinations))
    U_mat = np.zeros((len(a_values), len(s_values)))

    # Populate the U_mat matrix
    for s_idx, s in enumerate(s_combinations):
        vA = s[0]  # v_A from the first triplet (vA, u1A, u2A)
    
        for a_idx, a in enumerate(a_values):
            # Calculate the utility U(a, s) and store it in U_mat
            U_mat[a_idx, s_idx] = 1+a * (vA - 1)

    return a_values, a_strings, s_values, s_strings, U_mat

def setup_example_one_opp(scenario):
    # Try to retrieve the specific values, and fall back to general 'p1a' and 'p1b' if 'p1ad', 'p1bd', etc. are not present.
    p1a=scenario["p1a"]
    p1b=scenario["p1b"]
    p2a=scenario["p2a"]
    p2b=scenario["p2b"]
    cor12=scenario["cor12"]
    
    # Define the possible values for v, u1, u2
    u1_values = [0, 1, 10]
    u2_values = ["x", "y", "z"]
    u1_value_to_index = {val: idx for idx, val in enumerate(u1_values)}
    u2_value_to_index = {val: idx for idx, val in enumerate(u2_values)}

    # Generate all combinations for the observation vector (only u1 and u2 for both investments)
    o_combinations = list(itertools.product(u1_values, u2_values))
    # Enumerate them
    o_values = list(range(1, len(o_combinations) + 1))
    o_strings = [f"o1={o[0]} o2={o[1]}" for o in o_combinations]

    s_values=scenario["s_values"]
    # num_s_values=len(s_values)
    # ps = np.ones(num_s_values) / num_s_values
    sd, su = s_values  # Assuming s_values has two entries
    # sd * (1-p1)+su*p1=EV
    # Calculate prior probabilities such that the expected value is 0
    EV=1
    p1 = (EV-sd) / (su-sd)
    p2 = 1 - p1
    ps = np.array([p2, p1])
    #EU = -2*su*sd / (su-sd)
    #EU^2 = sd^2 su / (su-sd) - su^2 sd / (su-sd) = - sd su 
    #VU= -sd su (1+4 su sd /(su-sd)^2)

    # Define the conditional probabilities for u1 and u2 given v
    p_epsilon = 0.00001 #keep probabilities this distance from 0/1
    slight_bias_max = min(p1a - p_epsilon, 1 - p1a - p1b - p_epsilon)
    slight_bias=0.00 #slight bias to resolve ties and make middle state mean something
    slight_bias = min(slight_bias, slight_bias_max)
    u1_given_v = {
        sd: np.array([p1b, p1a-slight_bias, 1-p1a-p1b+slight_bias]),
        su: np.array([1-p1a-p1b-slight_bias,p1a+slight_bias,p1b])
    }

    u2_given_v = {
        sd: np.array([p2b, p2a, 1-p2a-p2b]),
        su: np.array([1-p2a-p2b-slight_bias,p2a+slight_bias,p2b])
    }

    # Initialize the pogs matrix with shape (len(o_combinations), len(s_combinations))
    pogs = np.zeros((len(o_combinations), len(s_values)))

    # Populate the pogs matrix
    for s_idx, s in enumerate(s_values):
        vA = s

        # Calculate the probability of observing o given s
        for o_idx, o in enumerate(o_combinations):
            o_u1A, o_u2A = o

            # Find the index of u1 in the relevant array
            u1A_idx = u1_value_to_index[o_u1A]
            u2A_idx = u2_value_to_index[o_u2A]
            
            # Conditional probability of u2 given u1
            if u1A_idx == u2A_idx:
                p_u2_given_u1 = cor12 + (1 - cor12) * u2_given_v[vA][u2A_idx]
            else:
                p_u2_given_u1 = (1 - cor12) * u2_given_v[vA][u2A_idx]
            
            pogs[o_idx, s_idx] = u1_given_v[vA][u1A_idx] * p_u2_given_u1

    # Initialize the marginals consistent with the conditionals
    po = compute_marginals_o(ps, pogs) 
    psgo = ((pogs * ps) / po[:, np.newaxis]).T
    psgo = psgo / np.sum(psgo, axis=0)

    scenario["o_values"] = o_values
    scenario["o_strings"] = o_strings 
    scenario["po"] = po
    scenario["ps"] = ps
    scenario["pogs"] = pogs 
    scenario["psgo"] = psgo 

    return scenario

# Function to compute marginals
def compute_marginals_o(ps, pogs):
    po = pogs @ ps
    po += np.finfo(float).eps
    po /= np.sum(po)

    return po

def generate_results(da=1,v_values=[2],n_samp_values=[0],gamma_values=[0],beta_ao_values=[1000],Ni=0,epsilon_conv=0.0001,use_IS=False,uscen_ids=slice(0, 5)):

    # Nested loop over v, beta_ao, and scenarios
    results = []
    for v in v_values:
        scenarios=setup_scenarios(da=da,v=v)
        for scenario in scenarios[uscen_ids]:
            scenario = setup_example_one_opp(scenario)
            for gamma in gamma_values:
                for beta_ao in beta_ao_values:
                    for n_samp in n_samp_values:

                        # First, run the scenario with sample_util=0
                        pa_0, pago, pags, performance_df, risk_charge, E_U_mat = BAiterations(
                            scenario, beta_ao=beta_ao, epsilon_conv=epsilon_conv,gamma=gamma,sample_util=0, compute_performance=True,performance_per_iteration=False, performance_as_dataframe=True, init_pago_uniformly=True
                        )
                        E_U = performance_df['E_U'].values[0]
                        Var_U = performance_df['Var_U'].values[0]
                        I_ao = performance_df['I_ao'].values[0]

                        if Ni>0:
                            pa_accumulated = np.zeros_like(pa_0)  # Accumulate pa values across Ni iterations
                            E_U_i = np.zeros(Ni)
                            Var_U_i = np.zeros(Ni)
                            np.random.seed(99)

                            for i in range(1, Ni + 1):
                                pa, pago, pags, performance_df, risk_charge, E_U_mat = BAiterations(
                                    scenario, beta_ao=beta_ao,epsilon_conv=epsilon_conv,sample_util=n_samp,gamma=gamma,use_IS=use_IS, compute_performance=True,performance_per_iteration=False, performance_as_dataframe=True, init_pago_uniformly=True
                                )
                                last_row = performance_df.apply(pd.Series.explode).iloc[[-1]]
                                E_U_i[i - 1] = last_row['E_U'].values[0]
                                Var_U_i[i - 1] = last_row['Var_U'].values[0]
                                # Accumulate pa values
                                pa_accumulated += pa

                            # Calculate mean and standard error for sample_util=3
                            mean_E_U = np.mean(E_U_i)
                            mean_Var_U = np.mean(Var_U_i)
                            if len(E_U_i) > 1:
                                std_dev = np.std(E_U_i, ddof=1)  # Sample standard deviation
                            else:
                                std_dev = 0  
                            n_samples = len(E_U_i)
                            standard_error = std_dev / np.sqrt(n_samples)

                            # Calculate the average pa across the Ni iterations
                            average_pa = pa_accumulated / Ni
                        else:
                            average_pa=[]
                            mean_E_U=[]
                            mean_Var_U=[]
                            standard_error=[]

                        # Store the results for both sample_util=0 and sample_util=3
                        results.append({
                            "v": v,"beta_ao": beta_ao,"gamma": gamma,
                            "scenario_id": scenario["id"],"scenario_name": scenario["name"],
                            "E_U": E_U, "Var_U": Var_U,"Vol_U": np.sqrt(Var_U), "I_ao":I_ao,
                            "n_samp": n_samp,
                            "E_U_sample": mean_E_U,"Var_U_sample":mean_Var_U,"Vol_U_sample": np.sqrt(mean_Var_U),
                            "standard_error_sample": standard_error,
                            "average_pa": average_pa,"mid_pa": np.sum(average_pa[1:-1]) if len(average_pa) > 2 else 0,
                            "pa_0": pa_0,"mid_pa_0": np.sum(pa_0[1:-1]) if len(pa_0) > 2 else 0
                        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by=['scenario_id'], ascending=[True])
    
    return res_df, scenarios