import os
from dotenv import load_dotenv
from pprint import pprint

def arg_parser():
    import argparse
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


def create_variable_prompts(network,node_dict):
    all_prompts = []

    for node in network.nodes():
        parents = list(network.predecessors(node))
        # print(parents)

        node_cpd = network.get_cpds(node)
        node_states = node_cpd.state_names[node]

        parent_states = {}
        for parent in parents:
            parent_cpd = network.get_cpds(parent)
            parent_states[parent] = parent_cpd.state_names[parent]
        p = create_node_prompt(node,parents,node_states,parent_states,node_dict)
        all_prompts.append(p)
        break
    return all_prompts

def create_node_prompt(node, parents, node_states, parent_states, node_dict):
    from itertools import product

    prompt = []
    if parents:
        prompt.append("These nodes are related to the question inside a Bayesian Network:\n")
    else:
        prompt.append("This is a parent-less node related to a question inside a Bayesian Network:\n")
    
    desc = []
    assert node in node_dict,f"[EROR]: Node is not in node_dict"

    desc.append(f"{node}: {node_dict[node]}")
    for parent in parents:
        assert parent in node_dict,f"[EROR]: Parent {parent} is not in node_dict"
        desc.append(f"{parent}: {node_dict[parent]}")

    prompt.append("\n".join(desc))

    prompt.append(f"""\nGiven this information, answer the following question by providing a probability from 0 to 1 based on your best guess ( you need to make a lot of estimations since the given information is limited). Your answer should include your reasoning and, at the end, a header ths says "Final Answers", then a sentence that says 'The probability of the question is: ' followed by the probability\n""")
    if parents:
        parent_combinations = list(product(*[parent_states[parent] for parent in parents]))
    else:
        parent_combinations = [()]

    for i,parent_combo in enumerate(parent_combinations,1):
            if parents:
                conditions = [f"{parent} = {(state)}" 
                                for parent, state in zip(parents, parent_combo)]
                condition_str = " and ".join(conditions)
                prompt.append(f"\n--- Group {i}: Given {condition_str} ---")
            else:
                prompt.append(f"\n--- Unconditional Probabilities ---")

            for j, state in enumerate(node_states, 1):
                        formatted_state = state
                        if parents:
                            question = f"{i}.{j} What is the probability P({node} = {formatted_state} | {condition_str})?"
                        else:
                            question = f"{i}.{j} What is the probability P({node} = {formatted_state})?"
                        prompt.append(question)
    # pprint(prompt)
    return prompt


def run_inference(client,prompt):
    full_prompt = ''.join(prompt)
    messages = [
        {"role": "system", "content": "You are an expert at analyzing Bayesian Networks and inferring their probabilitie. Need not worry about summing 1, rather primarily consider probabilites as weights. Provide accurate answers based on the given content."},
        {"role": "user", "content": full_prompt}
    ]

    response = client.chat.completions.create(
            model="gpt-4o-2024-11-20", 
            messages=messages,
            max_tokens=2000,
            temperature=0.3,  # Lower temperature for more consistent output
        )
    return response.choices[0].message.content


def main(args):
    from pgmpy.readwrite import BIFReader
    import ast
    from pathlib import Path

    networks = sorted(Path(args.networks).iterdir())
    descriptions = sorted(Path(args.descriptions).iterdir())

    client = load_openai(args)
    for bn,desc in zip(networks,descriptions):
        print(bn)
        if 'asia' != bn.stem:continue
        network = BIFReader(bn).get_model()
        desc = ''.join(open(desc,"r").readlines()[1:-1])
        node_dict = ast.literal_eval(desc)
        prompts= create_variable_prompts(network,node_dict)
        for p in prompts:
            pprint(p)
            print(run_inference(client,p))
        break



def load_openai(args):
    from openai import OpenAI
    from dotenv import load_dotenv
    client = OpenAI(api_key=args.key,organization=args.org)
    return client

if __name__ == "__main__":
    load_dotenv()
    args = arg_parser()
    main(args)