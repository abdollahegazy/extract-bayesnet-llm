import re
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from itertools import product

def create_individual_question_prompt(node, node_state, parents, parent_combo, node_dict):
    """Create a prompt for a single probability question"""

    prompt = []

    if parents:
        prompt.append("These nodes are related to the question inside a Bayesian Network:\n")
    else:
        prompt.append("This is a parent-less node related to a question inside a Bayesian Network:\n")
    
    # Add descriptions
    desc = [f"{node}: {node_dict[node]}"]
    for parent in parents:
        desc.append(f"{parent}: {node_dict[parent]}")
    
    prompt.append("\n".join(desc))
    prompt.append(f"""\nGiven this information, answer the following question by providing a probability from 0 to 1 based on your best guess. Your answer should include your reasoning and, at the end, a header that says "Final Answer", then a sentence that says 'The probability is: ' followed by the probability value.\n""")
    
    # Create the single question
    if parents:
        conditions = [f"{parent} = {state}" for parent, state in zip(parents, parent_combo)]
        condition_str = " and ".join(conditions)
        question = f"What is the probability P({node} = {node_state} | {condition_str})?"
    else:
        question = f"What is the probability P({node} = {node_state})?"
    
    prompt.append(question)
    
    return prompt

def parse_single_probability(response_text):
    """Parse a single probability from LLM response"""
    # Pattern 1: "The probability is: X"
    pattern1 = r"The probability is:\s*([0-9]*\.?[0-9]+)"
    matches1 = re.findall(pattern1, response_text, re.IGNORECASE)
    
    # Pattern 2: "probability is X" or "probability: X"
    pattern2 = r"probability\s*(?:is|:)\s*([0-9]*\.?[0-9]+)"
    matches2 = re.findall(pattern2, response_text, re.IGNORECASE)
    
    # Pattern 3: Numbers after "Final Answer"
    pattern3 = r"Final Answer.*?([0-9]*\.?[0-9]+)"
    matches3 = re.findall(pattern3, response_text, re.IGNORECASE | re.DOTALL)
    
    # Pattern 4: P(...) = X format
    pattern4 = r"P\([^)]+\)\s*=\s*([0-9]*\.?[0-9]+)"
    matches4 = re.findall(pattern4, response_text, re.IGNORECASE)
    
    # Use the most specific pattern that found results
    if matches1:
        return float(matches1[0])
    elif matches3:
        return float(matches3[0])
    elif matches4:
        return float(matches4[0])
    elif matches2:
        return float(matches2[0])
    
    return None

def collect_individual_responses(network, node_dict, client, interactive=True):
    """
    Collect responses for each individual probability question
    """
    all_probabilities = {}
    
    for node in network.nodes():
        print(f"\n{'='*60}")
        print(f"Processing node: {node}")
        print(f"{'='*60}")
        
        parents = list(network.predecessors(node))
        
        # Get states
        node_cpd = network.get_cpds(node)
        node_states = node_cpd.state_names[node]
        
        parent_states = {}
        for parent in parents:
            parent_cpd = network.get_cpds(parent)
            parent_states[parent] = parent_cpd.state_names[parent]
        
        # Generate all combinations
        if parents:
            parent_combinations = list(product(*[parent_states[parent] for parent in parents]))
        else:
            parent_combinations = [()]
        
        node_probabilities = {}
        
        # For each parent combination
        for parent_combo in parent_combinations:
            combo_probabilities = {}
            
            print(f"\nParent combination: {dict(zip(parents, parent_combo)) if parents else 'No parents'}")
            
            # Ask each state individually
            for node_state in node_states:
                print(f"  Asking about {node} = {node_state}")
                
                # Create individual prompt
                prompt = create_individual_question_prompt(
                    node, node_state, parents, parent_combo, node_dict
                )
                
                # Get response
                response = run_inference(client, prompt)
                
                # Parse probability
                probability = parse_single_probability(response)
                
                if probability is None and interactive:
                    print(f"Failed to parse response for {node}={node_state}")
                    print("Response:", response[:200] + "...")
                    manual_prob = input(f"Please enter probability for P({node}={node_state}|...): ")
                    try:
                        probability = float(manual_prob)
                    except:
                        probability = 0.5  # Default fallback
                elif probability is None:
                    print(f"Warning: Failed to parse probability for {node}={node_state}, using 0.5")
                    probability = 0.5
                
                # Ensure probability is in valid range
                probability = max(0.0, min(1.0, probability))
                
                combo_probabilities[node_state] = probability
                print(f"    Parsed probability: {probability}")
            
            # Store raw probabilities (before normalization)
            node_probabilities[parent_combo] = combo_probabilities
        
        all_probabilities[node] = {
            'probabilities': node_probabilities,
            'parents': parents,
            'parent_states': parent_states,
            'node_states': node_states
        }
    
    return all_probabilities

def normalize_probabilities_post_collection(all_probabilities):
    """
    Normalize probabilities after collection to sum to 1 for each parent combination
    """
    normalized_probabilities = {}
    
    for node, data in all_probabilities.items():
        node_probs = data['probabilities']
        normalized_node_probs = {}
        
        print(f"\nNormalizing probabilities for {node}:")
        
        for parent_combo, state_probs in node_probs.items():
            # Get raw probabilities
            raw_probs = [state_probs[state] for state in data['node_states']]
            total = sum(raw_probs)
            
            # Normalize
            if total > 0:
                normalized = [p / total for p in raw_probs]
            else:
                # If all zero, make uniform
                normalized = [1.0 / len(raw_probs)] * len(raw_probs)
            
            # Store normalized probabilities
            normalized_state_probs = {
                state: normalized[i] 
                for i, state in enumerate(data['node_states'])
            }
            
            normalized_node_probs[parent_combo] = normalized_state_probs
            
            # Print normalization info
            combo_str = dict(zip(data['parents'], parent_combo)) if data['parents'] else "No parents"
            print(f"  {combo_str}: {raw_probs} -> {normalized} (sum: {sum(normalized):.3f})")
        
        normalized_probabilities[node] = {
            'probabilities': normalized_node_probs,
            'parents': data['parents'],
            'parent_states': data['parent_states'],
            'node_states': data['node_states']
        }
    
    return normalized_probabilities

def construct_bayesian_network_from_individual_responses(original_network, normalized_probabilities):
    """
    Construct Bayesian Network from individually collected and normalized probabilities
    """
    # Create new network with same structure
    new_network = DiscreteBayesianNetwork()
    new_network.add_nodes_from(original_network.nodes())
    new_network.add_edges_from(original_network.edges())
    
    # Create CPDs for each node
    for node in new_network.nodes():
        node_data = normalized_probabilities[node]
        parents = node_data['parents']
        probabilities = node_data['probabilities']
        node_states = node_data['node_states']
        
        if parents:
            # Multi-dimensional CPD
            parent_states = node_data['parent_states']
            parent_combinations = list(product(*[parent_states[parent] for parent in parents]))
            
            # Build CPD values array
            cpd_values = []
            for state in node_states:
                state_probs = []
                for parent_combo in parent_combinations:
                    prob = probabilities[parent_combo][state]
                    state_probs.append(prob)
                cpd_values.append(state_probs)
            
            # Create CPD
            cpd = TabularCPD(
                variable=node,
                variable_card=len(node_states),
                values=cpd_values,
                evidence=parents,
                evidence_card=[len(parent_states[parent]) for parent in parents],
                state_names={node: node_states, **parent_states}
            )
        else:
            # Root node - no parents
            probs = [probabilities[()][state] for state in node_states]
            cpd_values = [[prob] for prob in probs]
            
            cpd = TabularCPD(
                variable=node,
                variable_card=len(node_states),
                values=cpd_values,
                state_names={node: node_states}
            )
        
        new_network.add_cpds(cpd)
    
    # Validate the network
    if new_network.check_model():
        print("✓ Constructed Bayesian Network is valid!")
    else:
        print("✗ Warning: Constructed network has issues")
    
    return new_network

def save_probabilities_to_file(all_probabilities, filename):
    """Save all collected probabilities to a file for later analysis"""
    import json
    
    # Convert to JSON-serializable format
    serializable_data = {}
    for node, data in all_probabilities.items():
        serializable_data[node] = {
            'probabilities': {
                str(combo): probs for combo, probs in data['probabilities'].items()
            },
            'parents': data['parents'],
            'parent_states': data['parent_states'],
            'node_states': data['node_states']
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Raw probabilities saved to: {filename}")


def count_configurations(model):
    parent_configs = 0
    full_configs = 0
    for node in model.nodes():
        cpd = model.get_cpds(node)
        child_card = len(cpd.state_names[node])
        parent_card_product = 1
        for parent in cpd.get_evidence():
            parent_card_product *= len(cpd.state_names[parent])
        parent_configs += parent_card_product
        full_configs += parent_card_product * child_card
    return parent_configs, full_configs


def main(args):
    from pgmpy.readwrite import BIFReader
    import ast
    from pathlib import Path

    networks = sorted(Path(args.networks).iterdir())
    descriptions = sorted(Path(args.descriptions).iterdir())

    client = load_openai(args)
    
    for bn, desc in zip(networks, descriptions):

        # if 'asia' != bn.stem: continue
        if Path(f'./EPK_4omini/{bn.stem}.bif').exists(): print(f"[SKIPPING] {bn.stem} b/c already done");continue
            
        # Load original network
        original_network = BIFReader(bn).get_model()
        n_conf = count_configurations(original_network)

        if n_conf[1]>1000: print(f"[SKIPPING] {bn.stem} b/c too large for now. (states:{n_conf[1]})");continue #too large

        
        # Load descriptions
        desc_text = ''.join(open(desc, "r").readlines()[1:-1])
        try:
            node_dict = ast.literal_eval(desc_text)
        except: print(f"[ERROR] {bn.stem} FAILED DUE TO AST LITERAL EVAL");continue
        
        n_nodes = len(original_network.nodes())
        print(f"Original network has {n_nodes} nodes")
        # if n_nodes >= 30: print(f"[SKIPPING] {bn.stem} b/c has too large for now. (nodes: {n_nodes})");continue
        print(f"\nProcessing: {bn} with {n_conf[1]} states and {n_nodes} nodes.")

        
        # Collect individual responses (no normalization bias)
        print("Collecting individual probability responses...")
        all_probabilities = collect_individual_responses(
            original_network, node_dict, client, interactive=True
        )
        
        # Save raw probabilities
        # save_probabilities_to_file(all_probabilities, f"raw_probabilities_{bn.stem}.json")
        
        # Normalize after collection
        print("\nNormalizing probabilities...")
        normalized_probabilities = normalize_probabilities_post_collection(all_probabilities)
        
        # Construct new network
        print("\nConstructing Bayesian Network...")
        constructed_network = construct_bayesian_network_from_individual_responses(
            original_network, normalized_probabilities
        )
        
        # from kl import kl_bn_local
        # div = kl_bn_local(original_network,constructed_network)
        # print(f"THIS IS THE KL DIV {div:.2f}")
        # Save the new network
        output_path = f"./EPK_4omini/{bn.stem}.bif"
        save_network(constructed_network, output_path)
        
        # Compare networks
        compare_networks(original_network, constructed_network)
        
        # break

def load_openai(args):
    from openai import OpenAI
    client = OpenAI(api_key=args.key, organization=args.org)
    return client

def run_inference(client, prompt):
    full_prompt = ''.join(prompt)
    messages = [
        {"role": "system", "content": "You are an expert at analyzing Bayesian Networks and inferring probabilities. Provide your best probability estimate and clearly state it at the end."},
        {"role": "user", "content": full_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", 
        messages=messages,
        max_tokens=1500,
        temperature=0.3,
    )
    return response.choices[0].message.content

def save_network(network, filename):
    """Save the constructed network"""
    from pgmpy.readwrite import BIFWriter
    writer = BIFWriter(network)
    writer.write_bif(filename)
    print(f"Network saved to: {filename}")

def compare_networks(original, constructed, num_samples=5):
    """Compare original vs constructed network with random queries"""
    print("\n" + "="*60)
    print("NETWORK COMPARISON")
    print("="*60)
    
    # Create inference objects
    orig_inference = VariableElimination(original)
    new_inference = VariableElimination(constructed)
    
    # Compare some random queries
    nodes = list(original.nodes())
    
    for i in range(num_samples):
        # Random query
        target = np.random.choice(nodes)
        
        try:
            # Simple marginal
            orig_result = orig_inference.query([target])
            new_result = new_inference.query([target])
            
            print(f"\nQuery {i+1}: P({target})")
            print("Original:", orig_result.values)
            print("LLM-built:", new_result.values)
            print("Difference:", np.abs(orig_result.values - new_result.values))
        except Exception as e:
            print(f"Query {i+1} failed: {e}")
def arg_parser():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Run EPK')

    parser.add_argument('--network_path', dest='networks', default="../data/filtered_networks",
                        help='Input test dataset CSV', type=str)
    parser.add_argument('--node_desc_path', dest='descriptions', default="../data/describe_nodes/data",
                        help='Input test dataset CSV', type=str)

  
    parser.add_argument('--model', dest='model', nargs='+',
                        default="gpt-4o-2024-11-20",
                        help="model name to run.")
    parser.add_argument('--maxattempt', dest='maxattempt', default=3,
                        help='Max retry attempts on error', type=int)
    
    parser.add_argument('--openaikey', dest='key', default=os.getenv("OPENAI_API_KEY"),
                        help='OpenAI API key', type=str)
    parser.add_argument('--openaiorg', dest='org', default=os.getenv("OPENAI_ORG_ID"),
                        help='OpenAI organization ID', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    args = arg_parser()
    main(args)