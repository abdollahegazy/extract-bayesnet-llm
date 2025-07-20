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
import re


class BnRepConverter:
    """
    Converter class to transform Bayesian networks from R's bnlearn format to Python's pgmpy compatible format
    """
    def __init__(self):
        self.bnlearn = importr('bnlearn')
        self.utils = importr('utils')
        print("R packages loaded successfully")

    
    def load_bnrep_network(self, network_name):
        try:
            robjects.r(f'library(bnRep)')
            robjects.r(f'data({network_name})')
            bn_r = robjects.r(network_name)
            return bn_r
        except:
            return None

    
    def extract_network_structure(self, bn_r):
        """
        Extract the DAG structure from R Bayesian network.
        
        Parameters:
        bn_r: R Bayesian network object
        
        Returns:
        list: List of edges as tuples
        """
        try:

            # Get the arcs from the network
            arcs_r = self.bnlearn.arcs(bn_r)
            print(arcs_r)
            arcs_array = np.array(arcs_r) #always 2n in length

            # Split into two halves (from and to nodes)
            n_edges = len(arcs_array) // 2
            from_nodes = arcs_array[:n_edges]
            to_nodes = arcs_array[n_edges:]
            
            # print(f"From nodes: {from_nodes}")
            # print(f"To nodes: {to_nodes}")
            
            # create edges by pairing corresponding elements
            edges = []
            for i in range(n_edges):
                parent = self.clean_state_names([from_nodes[i]])[0]
                child = self.clean_state_names([to_nodes[i]])[0]
                edges.append((parent, child))
            
            # print(f"Extracted edges: {edges}")
            # quit()
            return edges
            
        except Exception as e:
            print(f"Error extracting network structure: {e}")
            return []
    
    def extract_node_names(self, bn_r):
        """
        Extract node names from R Bayesian network.
        
        Parameters:
        bn_r: R Bayesian network object
        
        Returns:
        list: List of node names
        """
        try:
            nodes_r = self.bnlearn.nodes(bn_r)
            nodes = self.clean_state_names(list(pandas2ri.rpy2py(nodes_r)))
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
    
    def extract_cpds(self, bn_r,verbose=False):
        """
        Extract Conditional Probability Distributions from fitted R network.
        
        Parameters:
        bn_r: R Bayesian network object (should be fitted)
        
        Returns:
        dict: Dictionary of CPDs for each node
        """

        cpds = {}
        
        nodes = self.extract_node_names(bn_r)
        robjects.globalenv['fitted_bn'] = bn_r
        test_result = robjects.r('class(fitted_bn)')
        is_fitted = 'bn.fit' in str(test_result)
        assert is_fitted,"Fit model urself. Error"

            
        # Extract CPDs for each node
        for node in nodes:
            print(f"Extracting CPD for node: {node}")
            
            # Get the CPD table for this node
            robjects.r(f'node_cpd <- fitted_bn${node}')
            
            # Get the probability table
            prob_table = robjects.r(f'fitted_bn${node}$prob')
            
            # Get parent nodes for this node
            parents_r = robjects.r(f'fitted_bn${node}$parents')
            parents = list(parents_r) if parents_r != robjects.NULL else []
            
            # Get node states (levels)
            node_states_r = robjects.r(f'dimnames(fitted_bn${node}$prob)${node}')
            node_states = self.clean_state_names(list(node_states_r))

            if verbose:
                print(f"  Node: {node}")
                print(f"  Parents: {parents}")
                print(f"  States: {node_states}")
            
            if prob_table != robjects.NULL:
                # Convert probability table to numpy array
                prob_array = np.array(prob_table)

                if verbose:
                    print(f"  Probability array shape: {prob_array.shape}")
                    print(f"  Probability array: {prob_array}")
                
                # Create TabularCPD
                if len(parents) == 0:
                    # Root node (no parents)
                    cpd = TabularCPD(
                        variable=node,
                        variable_card=len(node_states),
                        values=prob_array.reshape(-1, 1),
                        state_names={node: node_states}
                    )
                else:  # Node with parents
                    # Get parent states
                    parent_states = {}
                    parent_cards = []
                    
                    for parent in parents:
                        parent_states_r = robjects.r(f'dimnames(fitted_bn${node}$prob)${parent}')
                        if parent_states_r != robjects.NULL:
                            p_states = self.clean_state_names(list(parent_states_r))
                            parent_states[parent] = p_states
                            parent_cards.append(len(p_states))

                    # Reshape probability array for pgmpy format
                    # pgmpy expects shape: (node_card, parent1_card * parent2_card * ...)
                    # if prob_array.ndim > 1:
                    #     # Flatten and reshape appropriately
                    #     prob_values = prob_array.reshape(len(node_states), -1)
                    # else:
                    #     prob_values = prob_array.reshape(-1, 1)

                    if prob_array.ndim > 2:
                        # print(f"DEBUG {node}: Original 3D shape: {prob_array.shape}")
                        # print(f"DEBUG {node}: Parent cards: {parent_cards}")
                        # print(f"DEBUG {node}: Expected final shape: ({len(node_states)}, {np.prod(parent_cards)})")
                        # 
                        # Show the original 3D array structure
                        # print(f"DEBUG {node}: Original array:")
                        # print(prob_array)
                        
                        # Try correct reshaping for R's column-major order
                        # R stores arrays column-major, so we need to transpose properly
                        prob_values = prob_array.reshape(len(node_states), -1, order='F')  # Fortran/column-major order
                        
                        # print(f"DEBUG {node}: After reshape: {prob_values.shape}")
                        # print(f"DEBUG {node}: Reshaped array:")
                        # print(prob_values)
                        # print(f"DEBUG {node}: Column sums: {[np.sum(prob_values[:, i]) for i in range(prob_values.shape[1])]}")
                    elif prob_array.ndim == 2:
                        prob_values = prob_array.reshape(len(node_states), -1)
                    else:
                        prob_values = prob_array.reshape(-1, 1)
                    
                    if verbose:
                        print(f"  Parent cards: {parent_cards}")
                        print(f"  Reshaped values: {prob_values.shape}")
                    
                    # Create state names dict
                    state_names = {node: node_states}
                    state_names.update(parent_states)
                    
                    cpd = TabularCPD(
                        variable=node,
                        variable_card=len(node_states),
                        values=prob_values,
                        evidence=parents,
                        evidence_card=parent_cards,
                        state_names=state_names
                    )
                
                cpds[node] = cpd
                print(f" CPD created successfully for node {node}")
                
            else:
                print(f" Warning: No probability table found for node {node}")
    
        return cpds
    
    def create_pgmpy_network(self, network_name, data=None):
        """
        Create a complete pgmpy BayesianNetwork from bnRep network.
        
        Parameters:
        network_name (str): Name of the network in bnRep
        data (DataFrame): Optional data for parameter learning
        
        Returns:
        BayesianNetwork: pgmpy Bayesian network object
        """
        # Load the R network
        bn_r = self.load_bnrep_network(network_name)

        if bn_r is None:
            return None
        
        # Extract structure
        edges = self.extract_network_structure(bn_r)
        nodes = self.extract_node_names(bn_r)
        
        nodes_in_edges = set()
        for edge in edges:
            nodes_in_edges.add(edge[0])  # parent
            nodes_in_edges.add(edge[1])  # child

        # Create pgmpy network
        model = DiscreteBayesianNetwork(edges)
        
        print(f"Created network with {len(nodes)} nodes and {len(edges)} edges")

        print(f"Nodes: {nodes}")
        print(f"Edges: {edges}")


        print("\nExtracting CPDs")
        cpds_dict = self.extract_cpds(bn_r, verbose=True)
        print(f"Successfully extracted {len(cpds_dict)} CPDs")

        for node, cpd in cpds_dict.items():
            if node in nodes_in_edges:
                try:
                    model.add_cpds(cpd)
                    print(f"Added CPD for {node}")
                except Exception as e:
                    print(f"Error adding CPD for {node}: {e}")
            else:
                print(f"WARNING: Skipping isolated node: {node}")
        
        # Check if model is valid
        try:
            if model.check_model():
                print("Model validation successful!")
            else:
                print("Model validation failed")
        except Exception as e:
            print(f"Model validation error: {e}")
            return None
            
        return model
    
    def save_network(self, model, filename):
        """
        Save the pgmpy network to file.
        """

        writer = BIFWriter(model)
        writer.write_bif(filename)

            
        print(f"Network saved to {filename}")
            

def convert_network(network_name, output_dir='./converted_networks'):
    """
    Convert a single network from bnRep to pgmpy format.
    """

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize converter
    converter = BnRepConverter()
    
    # Convert network
    model = converter.create_pgmpy_network(network_name)
    
    if model is not None:
        # Save in multiple formats
        base_filename = os.path.join(output_dir, network_name)
        converter.save_network(model, f"{base_filename}.bif")
        return model
    else:
        print(f"ERROR: Failed to convert network: {network_name}")
        return None

def convert_all_bnrep_networks(network_names,output_dir='./zero_ichi'):
    """
    Convert all available networks from bnRep to pgmpy format.
    """

    
    converted_networks = {}
    
    for network_name in network_names:
        print(f"\nConverting network: {network_name}")
        try:
            model = convert_network(network_name, output_dir)
            if model is not None:
                converted_networks[network_name] = model
                print(f"Successfully converted {network_name}")
            else:
                print(f"Failed to convert {network_name}")
        except Exception as e:
            print(f"Error converting {network_name}: {e}")


        robjects.r('rm(list = ls())')
        robjects.r('gc()')
    
    
    print(f"\nConversion complete. Successfully converted {len(converted_networks)} networks.")
    return converted_networks

