import numpy as np
import networkx as nx


import numpy as np

def are_neighboring_states(state1, state2):
    """
    Check if two states are neighbors.
    """
    r1, θ1, ṙ1, θ̇1 = state1
    r2, θ2, ṙ2, θ̇2 = state2
    if abs(r1 - r2) > 1:
        return False
    if abs(θ1 - θ2) > 1:
        return False
    if abs(ṙ1 - ṙ2) > 1:
        return False
    if abs(θ̇1 - θ̇2) > 1:
        return False
    return True

def state_index_to_tuple(state_index):
    """
    Convert a state index to a tuple (r, θ, ṙ, θ̇).
    """
    r = state_index // (3 * 2 * 2)  # 3 θ bins * 2 ṙ bins * 2 θ̇ bins
    θ = (state_index // (2 * 2)) % 3  # 2 ṙ bins * 2 θ̇ bins
    ṙ = (state_index // 2) % 2  # 2 θ̇ bins
    θ̇ = state_index % 2
    return (r, θ, ṙ, θ̇)

def obs_index_to_tuple(obs_index):
    """
    Convert an observation index to a tuple (r, θ).
    
    Parameters:
    - obs_index: Integer index of the observation (0 to 8).
    
    Returns:
    - Tuple (r, θ).
    """
    r = obs_index // 3  # 3 θ bins
    θ = obs_index % 3
    return (r, θ)

num_states = 36 # [r (3), θ(3), ṙ(2), θ̇(2)]
transition_prob = np.zeros((num_states, num_states))

#High probability of staying in the same state
for i in range(num_states):
    for j in range(num_states):
        if i == j:
            transition_prob[i][j] = 0.6  # High probability of staying
        elif are_neighboring_states(state_index_to_tuple(i), state_index_to_tuple(j)):
            transition_prob[i][j] = 0.3  # Higher probability for neighboring states
        else:
            transition_prob[i][j] = 0.1 / (num_states - 2)  # Lower probability for non-neighboring states
    transition_prob[i] /= np.sum(transition_prob[i])  # Normalize


num_observations = 9
emission_prob = np.zeros((num_states, num_observations))

# Define Gaussian-like emission probabilities
for state in range(num_states):
    state_r, state_θ = state_index_to_tuple(state)[:2]  # Extract r and θ from state
    for obs in range(num_observations):
        obs_r, obs_θ = obs_index_to_tuple(obs)  # Extract r and θ from observation
        # Compute probability based on Gaussian distribution
        emission_prob[state][obs] = np.exp(-0.5 * ((state_r - obs_r) ** 2 + (state_θ - obs_θ) ** 2))
    emission_prob[state] /= np.sum(emission_prob[state])  # Normalize probabilities

initial_prob = np.ones(num_states) / num_states  # Uniform distribution

####################################################################################################

class AdaptiveOrderHMM:
    def __init__(self, num_states=num_states, transition_prob=transition_prob, emission_prob=emission_prob, initial_prob=initial_prob):
        """
        Initialize AO-HMM.
        
        Parameters:
        - num_states: Number of states in the HMM.
        - transition_prob: Transition probability matrix (num_states x num_states).
        - emission_prob: Emission probability matrix (num_states x num_observations).
        - initial_prob: Initial state probability vector (num_states).
        """
        self.num_states = num_states
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.initial_prob = initial_prob
        self.active_states = set()  # Active states in the current time window
        self.neighbor_states = set()  # Neighbor states (1-hop or 2-hop)

    def refine_state(self, observations):
        """
        Refine the state sequence using AO-HMM.
        
        Parameters:
        - observations: List of observations (e.g., [r, θ] measurements).
        
        Returns:
        - refined_sequence: Refined state sequence.
        """
        # Step 1: Select active states and their neighbors
        self.active_states = self._select_active_states(observations)
        self.neighbor_states = self._get_neighbor_states(self.active_states)

        # Step 2: Refine the state sequence using Viterbi decoding
        refined_sequence = self._viterbi_decode(observations)
        return refined_sequence

    def _select_active_states(self, observations):
        """
        Select active states based on the observations.
        
        Parameters:
        - observations: List of observations.
        
        Returns:
        - active_states: Set of active states.
        """
        # Example: Select states with high emission probability for the observations
        active_states = set()
        for obs in observations:
            for state in range(self.num_states):
                if self.emission_prob[state][obs] > 0.5:  # Threshold for active states
                    active_states.add(state)
        return active_states

    def _get_neighbor_states(self, active_states):
        """
        Get neighbor states (1-hop or 2-hop) for the active states.
        
        Parameters:
        - active_states: Set of active states.
        
        Returns:
        - neighbor_states: Set of neighbor states.
        """
        neighbor_states = set()
        for state in active_states:
            # Add 1-hop neighbors
            neighbor_states.update(np.where(self.transition_prob[state] > 0)[0])
            # Add 2-hop neighbors
            for neighbor in np.where(self.transition_prob[state] > 0)[0]:
                neighbor_states.update(np.where(self.transition_prob[neighbor] > 0)[0])
        return neighbor_states

    def _viterbi_decode(self, observations):
        """
        Perform Viterbi decoding to find the most likely state sequence.
        
        Parameters:
        - observations: List of observations.
        
        Returns:
        - state_sequence: Most likely state sequence.
        """
        num_obs = len(observations)
        dp = np.zeros((self.num_states, num_obs))  # DP table
        path = np.zeros((self.num_states, num_obs), dtype=int)  # Path table

        # Initialize DP table
        for state in range(self.num_states):
            dp[state][0] = self.initial_prob[state] * self.emission_prob[state][observations[0]]

        # Fill DP table
        for t in range(1, num_obs):
            for state in range(self.num_states):
                max_prob = -1
                max_state = -1
                for prev_state in range(self.num_states):
                    prob = dp[prev_state][t - 1] * self.transition_prob[prev_state][state] * self.emission_prob[state][observations[t]]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = prev_state
                dp[state][t] = max_prob
                path[state][t] = max_state

        # Backtrack to find the most likely state sequence
        state_sequence = []
        last_state = np.argmax(dp[:, -1])
        state_sequence.append(last_state)
        for t in range(num_obs - 1, 0, -1):
            last_state = path[last_state][t]
            state_sequence.append(last_state)
        state_sequence.reverse()

        return state_sequence
    


class CPDA:
    def __init__(self, time_window=5):
        """
        Initialize CPDA (crossover path disambiguation algorithm).
        
        Parameters:
        - time_window: Number of time windows to consider for disambiguation.
        """
        self.time_window = time_window
        self.interaction_graph = nx.Graph()  # Interaction graph for path disambiguation

    def disambiguate_path(self, state_sequences):
        """
        Disambiguate paths using CPDA.
        
        Parameters:
        - state_sequences: List of state sequences from AO-HMM.
        
        Returns:
        - disambiguated_sequences: List of disambiguated state sequences.
        """
        # Step 1: Build interaction graph
        self._build_interaction_graph(state_sequences)

        # Step 2: Disambiguate paths
        disambiguated_sequences = self._apply_cpda(state_sequences)
        return disambiguated_sequences

    def _build_interaction_graph(self, state_sequences):
        """
        Build an interaction graph from the state sequences.
        
        Parameters:
        - state_sequences: List of state sequences.
        """
        for sequence in state_sequences:
            for i in range(len(sequence) - 1):
                self.interaction_graph.add_edge(sequence[i], sequence[i + 1])

    def _apply_cpda(self, state_sequences):
        """
        Apply CPDA to disambiguate paths.
        
        Parameters:
        - state_sequences: List of state sequences.
        
        Returns:
        - disambiguated_sequences: List of disambiguated state sequences.
        """
        disambiguated_sequences = []
        for sequence in state_sequences:
            if len(sequence) > 1:
                # Find the shortest path in the interaction graph
                disambiguated_sequence = nx.shortest_path(self.interaction_graph, source=sequence[0], target=sequence[-1])
                disambiguated_sequences.append(disambiguated_sequence)
            else:
                disambiguated_sequences.append(sequence)
        return disambiguated_sequences