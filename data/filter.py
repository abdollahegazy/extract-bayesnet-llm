#!/usr/bin/env python3
"""
Bayesian Network Filter Script

Filters converted BIF files based on the following criteria:
1. Only discrete CPT values (no continuous variables)
2. Networks with 50 or fewer nodes
3. Complete CPT information (no missing probabilities)

Based on the filtering criteria from the research paper.
"""

import os
import shutil
from pathlib import Path
import numpy as np
from pgmpy.readwrite import BIFReader
import pandas as pd
from collections import defaultdict

class BayesianNetworkFilter:
    """Filter Bayesian networks based on specified criteria."""
    
    def __init__(self, input_dir='./converted_networks', output_dir='./filtered_networks'):
        """
        Initialize the filter.
        
        Parameters:
        input_dir (str): Directory containing BIF files to filter
        output_dir (str): Directory to save filtered networks
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_networks': 0,
            'discrete_only': 0,
            'under_50_nodes': 0,
            'complete_cpts': 0,
            'final_filtered': 0,
            'failed_to_load': 0
        }
        
        # Store details about filtered networks
        self.network_details = []
        
    def load_network(self, bif_file):
        """
        Load a BIF file and return the model.
        
        Parameters:
        bif_file (Path): Path to BIF file
        
        Returns:
        model or None: Loaded pgmpy model or None if failed
        """
        try:
            reader = BIFReader(str(bif_file))
            model = reader.get_model()
            return model
        except Exception as e:
            print(f"  ✗ Failed to load {bif_file.name}: {e}")
            self.stats['failed_to_load'] += 1
            return None
    
    def is_discrete_only(self, model):
        """
        Check if all variables in the network are discrete.
        
        Parameters:
        model: pgmpy BayesianNetwork
        
        Returns:
        bool: True if all variables are discrete
        """
        try:
            # Check if all CPDs are TabularCPD (discrete)
            from pgmpy.factors.discrete import TabularCPD
            
            cpds = model.get_cpds()
            if not cpds:  # No CPDs
                return False
                
            for cpd in cpds:
                if not isinstance(cpd, TabularCPD):
                    return False
            
            # Additional check: ensure all variables have finite cardinality
            for node in model.nodes():
                try:
                    cardinality = model.get_cardinality(node)
                    if cardinality is None or cardinality <= 0:
                        return False
                except:
                    return False
            
            return True
            
        except Exception as e:
            print(f"    Warning: Could not check if network is discrete: {e}")
            return False
    
    def has_50_or_fewer_nodes(self, model):
        """
        Check if network has 50 or fewer nodes.
        
        Parameters:
        model: pgmpy BayesianNetwork
        
        Returns:
        bool: True if network has <= 50 nodes
        """
        return len(model.nodes()) <= 50
    
    def has_complete_cpts(self, model):
        """
        Check if all CPTs have complete probability information.
        """
        try:
            cpds = model.get_cpds()
            if not cpds:
                return False

            for cpd in cpds:
                # raw values (might be flat)
                flat = np.array(cpd.get_values(), dtype=float)
                # how many states does the child have?
                r = cpd.variable_card
                # how many parent‐configs?
                n = flat.size // r

                if flat.size != r * n:
                    return False  # something’s really weird

                # reshape into (r states × n configs)
                table = flat.reshape((r, n), order='F')

                # check for any NaNs or negatives
                if np.any(np.isnan(table)) or np.any(table < 0):
                    return False

                # for each config, states must sum to 1
                tol = 1e-8
                col_sums = table.sum(axis=0)  # sum over states
                if not np.allclose(col_sums, 1.0, atol=tol, rtol=tol):
                    bad = np.where(~np.isclose(col_sums, 1.0, atol=tol, rtol=tol))[0]
                    print(f"    Warning: CPD {cpd.variable} config(s) {bad.tolist()} sum to {col_sums[bad]}")
                    return False

            # finally, let pgmpy sanity‑check the whole model
            return model.check_model()
        except Exception:
            return False
                
 
    def get_network_info(self, model, network_name):
        """
        Extract detailed information about a network.
        
        Parameters:
        model: pgmpy BayesianNetwork
        network_name (str): Name of the network
        
        Returns:
        dict: Network information
        """
        try:
            info = {
                'name': network_name,
                'num_nodes': len(model.nodes()),
                'num_edges': len(model.edges()),
                'num_cpds': len(model.get_cpds()),
                'nodes': list(model.nodes()),
                'max_cardinality': max([model.get_cardinality(node) for node in model.nodes()]),
                'avg_cardinality': np.mean([model.get_cardinality(node) for node in model.nodes()]),
                'is_discrete': self.is_discrete_only(model),
                'under_50_nodes': self.has_50_or_fewer_nodes(model),
                'complete_cpts': self.has_complete_cpts(model)
            }
            return info
        except Exception as e:
            print(f"    Warning: Could not extract network info: {e}")
            return {
                'name': network_name,
                'error': str(e)
            }
    
    def filter_networks(self, verbose=True):
        """
        Filter all BIF files in input directory based on criteria.
        
        Parameters:
        verbose (bool): Print detailed progress information
        
        Returns:
        list: List of successfully filtered network names
        """
        bif_files = list(self.input_dir.glob('*.bif'))
        self.stats['total_networks'] = len(bif_files)
        
        if verbose:
            print(f"Found {len(bif_files)} BIF files to filter")
            print("=" * 60)
        
        filtered_networks = []
        
        for bif_file in bif_files:

            network_name = bif_file.stem

            # if network_name !='crypto':
                # continue
            
            if verbose:
                print(f"\nProcessing: {network_name}")
            
            # Load network
            model = self.load_network(bif_file)
            if model is None:
                continue
            
            # Get network information
            info = self.get_network_info(model, network_name)
            self.network_details.append(info)
            
            if verbose:
                print(f"  Nodes: {info.get('num_nodes', 'Unknown')}")
                print(f"  Edges: {info.get('num_edges', 'Unknown')}")
                print(f"  CPDs: {info.get('num_cpds', 'Unknown')}")
            
            # Apply filters
            passed_filters = True
            
            # Filter 1: Discrete only
            if not self.is_discrete_only(model):
                if verbose:
                    print("  ✗ Failed: Contains non-discrete variables")
                passed_filters = False
            else:
                self.stats['discrete_only'] += 1
                if verbose:
                    print("  ✓ Passed: All discrete variables")
            
            # Filter 2: 50 or fewer nodes
            if not self.has_50_or_fewer_nodes(model):
                if verbose:
                    print(f"  ✗ Failed: Too many nodes ({info.get('num_nodes', 'Unknown')} > 50)")
                passed_filters = False
            else:
                self.stats['under_50_nodes'] += 1
                if verbose:
                    print(f"  ✓ Passed: Node count acceptable ({info.get('num_nodes', 'Unknown')} <= 50)")
            
            # Filter 3: Complete CPTs
            if not self.has_complete_cpts(model):
                if verbose:
                    print("  ✗ Failed: Incomplete CPT information")
                passed_filters = False
            else:
                self.stats['complete_cpts'] += 1
                if verbose:
                    print("  ✓ Passed: Complete CPT information")
            
            # If passed all filters, copy to output directory
            if passed_filters:
                self.stats['final_filtered'] += 1
                filtered_networks.append(network_name)
                
                # Copy file to output directory
                output_file = self.output_dir / bif_file.name
                shutil.copy2(bif_file, output_file)
                
                if verbose:
                    print(f"  ✓ ACCEPTED: Copied to {output_file}")
            else:
                if verbose:
                    print(f"  ✗ REJECTED")
        
        return filtered_networks
    
    def print_summary(self):
        """Print filtering summary statistics."""
        print("\n" + "=" * 60)
        print("FILTERING SUMMARY")
        print("=" * 60)
        
        print(f"Total networks processed: {self.stats['total_networks']}")
        print(f"Failed to load: {self.stats['failed_to_load']}")
        print(f"Successfully loaded: {self.stats['total_networks'] - self.stats['failed_to_load']}")
        print()
        
        print("Filter Results:")
        print(f"  ✓ Discrete only: {self.stats['discrete_only']}")
        print(f"  ✓ ≤50 nodes: {self.stats['under_50_nodes']}")
        print(f"  ✓ Complete CPTs: {self.stats['complete_cpts']}")
        print()
        
        print(f"FINAL FILTERED NETWORKS: {self.stats['final_filtered']}")
        
        if self.stats['total_networks'] > 0:
            percentage = (self.stats['final_filtered'] / self.stats['total_networks']) * 100
            print(f"Success rate: {percentage:.1f}%")
    
    def save_detailed_report(self, filename='network_filtering_report.csv'):
        """
        Save detailed information about all networks to CSV.
        
        Parameters:
        filename (str): Output CSV filename
        """
        if self.network_details:
            df = pd.DataFrame(self.network_details)
            report_path = self.output_dir / filename
            df.to_csv(report_path, index=False)
            print(f"\nDetailed report saved to: {report_path}")
        else:
            print("No network details to save.")
    
    def analyze_filtered_networks(self):
        """Analyze the characteristics of filtered networks."""
        filtered_details = [info for info in self.network_details 
                          if info.get('is_discrete', False) and 
                             info.get('under_50_nodes', False) and 
                             info.get('complete_cpts', False)]
        
        if not filtered_details:
            print("No networks passed all filters.")
            return
        
        print("\n" + "=" * 60)
        print("FILTERED NETWORKS ANALYSIS")
        print("=" * 60)
        
        node_counts = [info['num_nodes'] for info in filtered_details]
        edge_counts = [info['num_edges'] for info in filtered_details]
        max_cards = [info['max_cardinality'] for info in filtered_details]
        
        print(f"Number of filtered networks: {len(filtered_details)}")
        print(f"Node count - Min: {min(node_counts)}, Max: {max(node_counts)}, Avg: {np.mean(node_counts):.1f}")
        print(f"Edge count - Min: {min(edge_counts)}, Max: {max(edge_counts)}, Avg: {np.mean(edge_counts):.1f}")
        print(f"Max cardinality - Min: {min(max_cards)}, Max: {max(max_cards)}, Avg: {np.mean(max_cards):.1f}")


def main():
    """Main function to run the filtering process."""
    print("Bayesian Network Filtering Tool")
    print("Filtering criteria:")
    print("1. Discrete CPT values only")
    print("2. Networks with ≤50 nodes")
    print("3. Complete CPT information")
    print()
    
    # Initialize filter
    filter_tool = BayesianNetworkFilter(
        input_dir='./zero_ichi',
        output_dir='./temp'
    )
    
    # Run filtering
    filtered_networks = filter_tool.filter_networks(verbose=True)
    
    # Print summary
    filter_tool.print_summary()
    
    # Analyze filtered networks
    filter_tool.analyze_filtered_networks()
    
    # Save detailed report
    filter_tool.save_detailed_report()
    
    print(f"\nFiltered networks saved to: {filter_tool.output_dir}")
    print(f"Network names: {filtered_networks}")


if __name__ == "__main__":
    main()