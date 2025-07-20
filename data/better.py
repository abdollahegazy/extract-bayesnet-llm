import warnings
warnings.filterwarnings("ignore", module="rpy2")

import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import BIFReader, BIFWriter
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import os
import json
import re



class ImprovedBnRepConverter:
    """
    Improved converter class with better error handling and CPD extraction.
    """
    
    def __init__(self):
        try:
            self.bnlearn = importr('bnlearn')
            self.utils = importr('utils')
            print("R packages loaded successfully")
        except Exception as e:
            print(f"Error loading R packages: {e}")
            raise
    
    def load_bnrep_network(self, network_name):
        """Load a network from bnRep dataset."""
        try:
            robjects.r(f'library(bnRep)')
            robjects.r(f'data({network_name})')
            bn_r = robjects.r(network_name)
            return bn_r
        except Exception as e:
            print(f"Error loading network {network_name}: {e}")
            return None
    
    def extract_network_structure(self, bn_r):
        """Extract DAG structure from R Bayesian network."""
        try:
            arcs_r = self.bnlearn.arcs(bn_r)
            arcs_array = np.array(arcs_r)
            
            # Handle empty networks
            if len(arcs_array) == 0:
                return []
            
            # Split into from/to nodes
            n_edges = len(arcs_array) // 2
            from_nodes = arcs_array[:n_edges]
            to_nodes = arcs_array[n_edges:]
            
            edges = [(from_nodes[i], to_nodes[i]) for i in range(n_edges)]
            return edges
            
        except Exception as e:
            print(f"Error extracting network structure: {e}")
            return []
    
    def extract_node_names(self, bn_r):
        """Extract node names from R Bayesian network."""
        try:
            nodes_r = self.bnlearn.nodes(bn_r)
            nodes = list(pandas2ri.rpy2py(nodes_r))
            return nodes
        except Exception as e:
            print(f"Error extracting node names: {e}")
            return []
    
    def clean_state_names(self, states):
        """Clean state names so they're single, valid tokens (no commas or spaces)."""
        cleaned = []
        for state in states:
            state_str = str(state).strip()
            # Replace any sequence of non-word chars (including spaces, commas, parens) with "_"
            state_str = re.sub(r'[^\w\-_.]+', '_', state_str)
            # Collapse possible leading/trailing underscores
            state_str = state_str.strip('_')
            # Fallback if empty
            if not state_str:
                state_str = f"state_{len(cleaned)}"
            cleaned.append(state_str)
        return cleaned
    
    def extract_cpds(self, bn_r, verbose=False):
        """
        Extract CPDs with improved error handling and debugging.
        """
        cpds = {}
        
        try:
            nodes = self.extract_node_names(bn_r)
            
            # Store network in R environment
            robjects.globalenv['fitted_bn'] = bn_r
            
            # Check if fitted
            test_result = robjects.r('class(fitted_bn)')
            is_fitted = 'bn.fit' in str(test_result)
            
            if not is_fitted:
                print("Network is not fitted. Cannot extract CPDs.")
                return cpds
            
            print(f"Extracting CPDs for {len(nodes)} nodes...")
            
            for node in nodes:
                try:
                    if verbose:
                        print(f"\nProcessing node: {node}")
                    
                    # Get probability table
                    prob_table = robjects.r(f'fitted_bn${node}$prob')
                    if prob_table == robjects.NULL:
                        print(f"  Warning: No probability table for {node}")
                        continue
                    
                    # Get parents
                    parents_r = robjects.r(f'fitted_bn${node}$parents')
                    parents = list(parents_r) if parents_r != robjects.NULL else []
                    
                    # Get node states
                    try:
                        node_states_r = robjects.r(f'dimnames(fitted_bn${node}$prob)${node}')
                        if node_states_r != robjects.NULL:
                            node_states = self.clean_state_names(list(node_states_r))
                        else:
                            # Fallback: try to get from cardinality
                            cardinality = robjects.r(f'dim(fitted_bn${node}$prob)[1]')[0]
                            node_states = [f"state_{i}" for i in range(int(cardinality))]
                    except:
                        # Final fallback
                        node_states = ["state_0", "state_1"]
                    
                    if verbose:
                        print(f"  Parents: {parents}")
                        print(f"  States: {node_states}")
                    
                    # Convert probability table
                    prob_array = np.array(prob_table)
                    
                    if verbose:
                        print(f"  Raw prob array shape: {prob_array.shape}")
                        print(f"  Raw prob array:\n{prob_array}")
                    
                    # Handle different cases
                    if len(parents) == 0:
                        # Root node
                        if prob_array.ndim == 1:
                            values = prob_array.reshape(-1, 1)
                        else:
                            values = prob_array
                        
                        cpd = TabularCPD(
                            variable=node,
                            variable_card=len(node_states),
                            values=values,
                            state_names={node: node_states}
                        )
                        
                    else:
                        # Node with parents
                        parent_states = {}
                        parent_cards = []
                        
                        for parent in parents:
                            try:
                                parent_states_r = robjects.r(f'dimnames(fitted_bn${node}$prob)${parent}')
                                if parent_states_r != robjects.NULL:
                                    p_states = self.clean_state_names(list(parent_states_r))
                                else:
                                    # Try to get from the parent's own CPD
                                    try:
                                        parent_states_r = robjects.r(f'dimnames(fitted_bn${parent}$prob)${parent}')
                                        p_states = self.clean_state_names(list(parent_states_r))
                                    except:
                                        p_states = ["state_0", "state_1"]
                                
                                parent_states[parent] = p_states
                                parent_cards.append(len(p_states))
                                
                            except Exception as e:
                                if verbose:
                                    print(f"    Warning: Could not get states for parent {parent}: {e}")
                                # Default to binary
                                parent_states[parent] = ["state_0", "state_1"]
                                parent_cards.append(2)
                        
                        if verbose:
                            print(f"  Parent cards: {parent_cards}")
                            print(f"  Parent states: {parent_states}")
                        
                        # Reshape probability values
                        # Expected shape: (node_card, total_parent_combinations)
                        expected_parent_combos = np.prod(parent_cards) if parent_cards else 1
                        
                        if prob_array.ndim == 1:
                            # 1D array - might need reshaping
                            if len(prob_array) == len(node_states) * expected_parent_combos:
                                values = prob_array.reshape(len(node_states), expected_parent_combos)
                            else:
                                print(f"    Error: Unexpected 1D array size. Expected {len(node_states) * expected_parent_combos}, got {len(prob_array)}")
                                continue
                        else:
                            # Multi-dimensional array
                            if prob_array.shape[0] == len(node_states):
                                # Correct first dimension
                                values = prob_array.reshape(len(node_states), -1)
                            else:
                                # Try transposing
                                prob_array_t = prob_array.T
                                if prob_array_t.shape[0] == len(node_states):
                                    values = prob_array_t.reshape(len(node_states), -1)
                                else:
                                    print(f"    Error: Cannot match array shape {prob_array.shape} to node cardinality {len(node_states)}")
                                    continue
                        
                        if verbose:
                            print(f"  Final values shape: {values.shape}")
                            print(f"  Expected shape: ({len(node_states)}, {expected_parent_combos})")
                        
                        # Verify shape matches expectations
                        if values.shape != (len(node_states), expected_parent_combos):
                            print(f"    Error: Shape mismatch. Got {values.shape}, expected ({len(node_states)}, {expected_parent_combos})")
                            continue
                        
                        # Create state names dict
                        state_names = {node: node_states}
                        state_names.update(parent_states)
                        
                        cpd = TabularCPD(
                            variable=node,
                            variable_card=len(node_states),
                            values=values,
                            evidence=parents,
                            evidence_card=parent_cards,
                            state_names=state_names
                        )
                    
                    # Validate CPD
                    try:
                        # Check if probabilities sum to 1
                        if len(parents) == 0:
                            prob_sum = np.sum(cpd.values)
                            if not np.isclose(prob_sum, 1.0, rtol=1e-10):
                                print(f"    Warning: Root node probabilities don't sum to 1: {prob_sum}")
                        else:
                            for col in range(cpd.values.shape[1]):
                                col_sum = np.sum(cpd.values[:, col])
                                if not np.isclose(col_sum, 1.0, rtol=1e-10):
                                    print(f"    Warning: Column {col} probabilities don't sum to 1: {col_sum}")
                        
                        cpds[node] = cpd
                        print(f"  ✓ CPD created successfully for {node}")
                        
                    except Exception as validation_error:
                        print(f"    ✗ CPD validation failed for {node}: {validation_error}")
                        continue
                        
                except Exception as node_error:
                    print(f"  ✗ Error processing node {node}: {node_error}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    continue
            
            print(f"Successfully extracted {len(cpds)} out of {len(nodes)} CPDs")
            return cpds
            
        except Exception as e:
            print(f"Error in CPD extraction: {e}")
            import traceback
            traceback.print_exc()
            return cpds
    
    def create_pgmpy_network(self, network_name, verbose=False):
        """Create complete pgmpy BayesianNetwork with better error handling."""
        
        print(f"\n{'='*50}")
        print(f"Converting network: {network_name}")
        print(f"{'='*50}")
        
        # Load R network
        bn_r = self.load_bnrep_network(network_name)
        if bn_r is None:
            return None
        
        # Extract structure
        edges = self.extract_network_structure(bn_r)
        nodes = self.extract_node_names(bn_r)
        
        if not nodes:
            print("No nodes found in network")
            return None
        
        print(f"Network structure: {len(nodes)} nodes, {len(edges)} edges")
        if verbose:
            print(f"Nodes: {nodes}")
            print(f"Edges: {edges}")
        
        # Create network
        try:
            model = DiscreteBayesianNetwork(edges)
        except Exception as e:
            print(f"Error creating network structure: {e}")
            return None
        
        # Extract and add CPDs
        try:
            cpds_dict = self.extract_cpds(bn_r, verbose=verbose)
            
            if not cpds_dict:
                print("No CPDs extracted")
                return None
            
            # Add CPDs to model
            added_cpds = 0
            for node, cpd in cpds_dict.items():
                try:
                    model.add_cpds(cpd)
                    added_cpds += 1
                    if verbose:
                        print(f"✓ Added CPD for {node}")
                except Exception as e:
                    print(f"✗ Error adding CPD for {node}: {e}")
            
            print(f"Added {added_cpds} CPDs to model")
            
            # Validate model
            try:
                if model.check_model():
                    print("✓ Model validation successful!")
                    return model
                else:
                    print("✗ Model validation failed")
                    return None
            except Exception as e:
                print(f"Model validation error: {e}")
                return None
                
        except Exception as e:
            print(f"Error in CPD processing: {e}")
            return None
    
    def save_network(self, model, filename):
        """Save network with better error handling."""
        try:
            writer = BIFWriter(model)
            writer.write_bif(filename)
            print(f"✓ Network saved to {filename}")
            return True
        except Exception as e:
            print(f"✗ Error saving network to {filename}: {e}")
            return False

def convert_network_improved(network_name, output_dir='./temp', verbose=False):
    """Convert single network with improved error handling."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    converter = ImprovedBnRepConverter()
    model = converter.create_pgmpy_network(network_name, verbose=verbose)
    
    if model is not None:
        filename = os.path.join(output_dir, f"{network_name}.bif")
        if converter.save_network(model, filename):
            return model
    
    print(f"✗ Failed to convert {network_name}")
    return None

def convert_all_improved(network_names, output_dir='./temp', verbose=False):
    """Convert all networks with improved error handling."""
    
    converted_networks = {}
    failed_networks = []
    
    for i, network_name in enumerate(network_names, 1):
        print(f"\n[{i}/{len(network_names)}] Converting: {network_name}")
        
        try:
            model = convert_network_improved(network_name, output_dir, verbose=verbose)
            if model is not None:
                converted_networks[network_name] = model
                print(f"✓ Successfully converted {network_name}")
            else:
                failed_networks.append(network_name)
                print(f"✗ Failed to convert {network_name}")
                
        except Exception as e:
            failed_networks.append(network_name)
            print(f"✗ Error converting {network_name}: {e}")
        
        # Clean R environment
        try:
            robjects.r('rm(list = ls())')
            robjects.r('gc()')
        except:
            pass
    
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total attempted: {len(network_names)}")
    print(f"Successfully converted: {len(converted_networks)}")
    print(f"Failed: {len(failed_networks)}")
    
    if failed_networks:
        print(f"\nFailed networks: {failed_networks}")
    
    return converted_networks

# Example usage
if __name__ == "__main__":
    # Test with a single network first
    # network_name = "asia"
    # model = convert_network_improved(network_name, verbose=True)
    
    # if model:
    #     print(f"Successfully converted {network_name}!")

    test_networks = ["fire", "coral3", "case", "covid2"]
    results = convert_all_improved(test_networks, verbose=True)