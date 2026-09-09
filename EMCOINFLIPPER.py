import numpy as np
from scipy.stats import binom

def em_coin_flipping_converge(tolerance=1e-6):
    # The dataset: (Heads, Tails) for each of the 5 coin tossing experiments
    experiments = [
        (5, 5), 
        (9, 1), 
        (8, 2), 
        (4, 6), 
        (7, 3)
    ]
    
    # Initial parameters assigned for the first iteration
    theta_a = 0.60
    theta_b = 0.50
    
    iteration_count = 0
    
    while True:
        iteration_count += 1
        
        # Initialize expected counts for the E-step
        expected_heads_a, expected_tails_a = 0.0, 0.0
        expected_heads_b, expected_tails_b = 0.0, 0.0
        
        # --- E-STEP (Expectation) ---
        for heads, tails in experiments:
            n = heads + tails
            
            # Calculate the likelihood of seeing this outcome from each coin
            likelihood_a = binom.pmf(heads, n, theta_a)
            likelihood_b = binom.pmf(heads, n, theta_b)
            
            # Normalize to get the responsibilities 
            prob_a = likelihood_a / (likelihood_a + likelihood_b)
            prob_b = likelihood_b / (likelihood_a + likelihood_b)
            
            # Distribute the actual observed heads and tails
            expected_heads_a += prob_a * heads
            expected_tails_a += prob_a * tails
            
            expected_heads_b += prob_b * heads
            expected_tails_b += prob_b * tails
            
        # --- M-STEP (Maximization) ---
        # Calculate new parameters
        new_theta_a = expected_heads_a / (expected_heads_a + expected_tails_a)
        new_theta_b = expected_heads_b / (expected_heads_b + expected_tails_b)
        
        # --- Convergence Check ---
        # If the change in both parameters is less than the tolerance, we have converged
        if abs(new_theta_a - theta_a) < tolerance and abs(new_theta_b - theta_b) < tolerance:
            theta_a = new_theta_a
            theta_b = new_theta_b
            break
            
        # Otherwise, update parameters and continue
        theta_a = new_theta_a
        theta_b = new_theta_b

    # Conclusion printed only after convergence is reached
    print("--- CONVERGENCE REACHED ---")
    print(f"Algorithm successfully converged after {iteration_count} iterations.")
    print(f"Final estimated bias for Coin A (Theta_A) : {theta_a:.4f}")
    print(f"Final estimated bias for Coin B (Theta_B) : {theta_b:.4f}")

if __name__ == '__main__':
    em_coin_flipping_converge()