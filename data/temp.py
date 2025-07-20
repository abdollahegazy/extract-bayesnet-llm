#!/usr/bin/env python3
"""
BIF File Diagnostic Tool

Diagnoses issues with BIF files to understand why CPDs are malformed.
"""

import numpy as np
from pgmpy.readwrite import BIFReader
import os
from pathlib import Path

def diagnose_bif_file(bif_path, verbose=True):
    """
    Diagnose a specific BIF file to understand CPD issues.
    
    Parameters:
    bif_path (str): Path to BIF file
    verbose (bool): Print detailed diagnostic info
    
    Returns:
    dict: Diagnostic results
    """
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {Path(bif_path).name}")
    print(f"{'='*60}")
    
    results = {
        'file': bif_path,
        'loaded': False,
        'nodes': [],
        'cpd_issues': [],
        'shape_issues': [],
        'sum_issues': []
    }
    
    try:
        # Load the model
        reader = BIFReader(bif_path)
        model = reader.get_model()
        results['loaded'] = True
        
        print(f"✓ Successfully loaded BIF file")
        print(f"  Nodes: {list(model.nodes())}")
        print(f"  Edges: {list(model.edges())}")
        
        results['nodes'] = list(model.nodes())
        
        # Check each CPD
        cpds = model.get_cpds()
        print(f"\nAnalyzing {len(cpds)} CPDs:")
        
        for cpd in cpds:
            node = cpd.variable
            print(f"\n--- CPD for {node} ---")
            
            # Get CPD details
            values = cpd.values
            parents = cpd.variables[1:] if len(cpd.variables) > 1 else []
            states = cpd.state_names[node]
            
            print(f"  Variable: {node}")
            print(f"  Parents: {parents}")
            print(f"  States: {states}")
            print(f"  Values shape: {values.shape}")
            print(f"  Values dtype: {values.dtype}")
            
            if verbose:
                print(f"  Raw values:\n{values}")
            
            # Check for shape issues
            expected_rows = len(states)
            if values.shape[0] != expected_rows:
                issue = f"Shape mismatch: expected {expected_rows} rows, got {values.shape[0]}"
                print(f"  ⚠️  {issue}")
                results['shape_issues'].append({'node': node, 'issue': issue})
            
            # Check for sum issues
            if values.ndim == 1:
                # Root node
                total_sum = np.sum(values)
                print(f"  Total sum: {total_sum}")
                if not np.isclose(total_sum, 1.0, rtol=1e-3, atol=1e-3):
                    issue = f"Root node sum = {total_sum} ≠ 1.0"
                    print(f"  ❌ {issue}")
                    results['sum_issues'].append({'node': node, 'issue': issue})
                else:
                    print(f"  ✓ Root node sum OK")
            else:
                # Node with parents
                print(f"  Column sums:")
                all_sums_ok = True
                for j in range(values.shape[1]):
                    col_sum = np.sum(values[:, j])
                    print(f"    Column {j}: {col_sum}")
                    if not np.isclose(col_sum, 1.0, rtol=1e-3, atol=1e-3):
                        issue = f"Column {j} sum = {col_sum} ≠ 1.0"
                        print(f"    ❌ {issue}")
                        results['sum_issues'].append({'node': node, 'issue': issue})
                        all_sums_ok = False
                
                if all_sums_ok:
                    print(f"  ✓ All column sums OK")
            
            # Check for NaN or negative values
            if np.any(np.isnan(values)):
                issue = "Contains NaN values"
                print(f"  ❌ {issue}")
                results['cpd_issues'].append({'node': node, 'issue': issue})
            
            if np.any(values < 0):
                issue = "Contains negative values"
                print(f"  ❌ {issue}")
                results['cpd_issues'].append({'node': node, 'issue': issue})
        
        # Try model validation
        print(f"\n--- Model Validation ---")
        try:
            if model.check_model():
                print("✓ Model validation passed")
            else:
                print("❌ Model validation failed")
                results['cpd_issues'].append({'node': 'model', 'issue': 'Failed model validation'})
        except Exception as e:
            print(f"❌ Model validation error: {e}")
            results['cpd_issues'].append({'node': 'model', 'issue': f'Validation error: {e}'})
        
    except Exception as e:
        print(f"❌ Failed to load BIF file: {e}")
        results['cpd_issues'].append({'node': 'file', 'issue': f'Load error: {e}'})
    
    return results

def diagnose_raw_bif_text(bif_path):
    """
    Analyze the raw BIF text to see if probabilities look correct.
    """
    print(f"\n--- Raw BIF Text Analysis ---")
    
    try:
        with open(bif_path, 'r') as f:
            content = f.read()
        
        # Find probability sections
        import re
        prob_sections = re.findall(r'probability\s*\([^}]+\}', content, re.DOTALL)
        
        for i, section in enumerate(prob_sections):
            print(f"\nProbability section {i+1}:")
            print(section)
            
            # Extract numbers from the section
            numbers = re.findall(r'\d+\.?\d*', section)
            if len(numbers) > 1:
                # Try to find probability values (skip variable names)
                prob_nums = []
                for num in numbers:
                    try:
                        val = float(num)
                        if 0 <= val <= 1:  # Likely a probability
                            prob_nums.append(val)
                    except:
                        continue
                
                if prob_nums:
                    print(f"  Extracted probabilities: {prob_nums}")
                    
                    # Check if they form valid probability rows
                    if len(prob_nums) % 2 == 0:  # Assuming binary variables
                        rows = [prob_nums[i:i+2] for i in range(0, len(prob_nums), 2)]
                        print(f"  Probability rows: {rows}")
                        for j, row in enumerate(rows):
                            row_sum = sum(row)
                            print(f"    Row {j} sum: {row_sum}")
    
    except Exception as e:
        print(f"Error reading raw BIF: {e}")

def diagnose_multiple_files(directory):
    """
    Diagnose multiple BIF files and summarize issues.
    """
    bif_files = list(Path(directory).glob('*.bif'))
    
    print(f"\nDiagnosing {len(bif_files)} BIF files in {directory}")
    print(f"{'='*80}")
    
    all_results = []
    
    for bif_file in bif_files[:5]:  # Limit to first 5 for testing
        result = diagnose_bif_file(str(bif_file), verbose=False)
        all_results.append(result)
        
        # Also check raw text
        diagnose_raw_bif_text(str(bif_file))
    
    # Summarize issues
    print(f"\n{'='*80}")
    print("SUMMARY OF ISSUES")
    print(f"{'='*80}")
    
    loaded_count = sum(1 for r in all_results if r['loaded'])
    print(f"Successfully loaded: {loaded_count}/{len(all_results)}")
    
    for result in all_results:
        file_name = Path(result['file']).name
        if result['sum_issues']:
            print(f"\n{file_name}:")
            for issue in result['sum_issues']:
                print(f"  ❌ {issue['node']}: {issue['issue']}")

def main():
    """Main diagnostic function."""
    # Test with a specific problematic file
    test_file = "./converted_networks/crypto.bif"  # Adjust path as needed
    
    if os.path.exists(test_file):
        diagnose_bif_file(test_file, verbose=True)
    else:
        print(f"Test file {test_file} not found.")
        print("Testing with files in ./converted_networks/")
        diagnose_multiple_files("./converted_networks")

if __name__ == "__main__":
    main()