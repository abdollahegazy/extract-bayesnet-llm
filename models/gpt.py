import os
import time
import argparse
import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv

def arg_parser():
    parser = argparse.ArgumentParser(description='Run zero-shot LLMs for Bayesian inference')
    parser.add_argument('--testdataset', dest='testdataset', default="../data/BLInD.csv",
                        help='Input test dataset CSV', type=str)
    parser.add_argument('--outputdataset', dest='outputdataset', default="../data/test/",
                        help='Folder to save results', type=str)
    parser.add_argument('--openaikey', dest='openaikey', default=os.getenv("OPENAI_API_KEY"),
                        help='OpenAI API key', type=str)
    parser.add_argument('--openaiorg', dest='openaiorg', default=os.getenv("OPENAI_ORG_ID"),
                        help='OpenAI organization ID', type=str)
    parser.add_argument('--method', dest='method', default="BQA",
                        choices=["BQA", "COT"], help='Inference method')
    parser.add_argument('--samplenum', dest='samplenum', default=915,
                        help='Number of samples to process', type=int)
    parser.add_argument('--model', dest='model', nargs='+',
                        default="gpt-4o-2024-11-20",
                        help="model name to run.")
    parser.add_argument('--maxattempt', dest='maxattempt', default=3,
                        help='Max retry attempts on error', type=int)
    parser.add_argument('--CLADDER', dest='CLADDER', action='store_true',
                        help='Use CLADDER dataset variant')
    return parser.parse_args()


def main():
    ...


if __name__ == "__main__":
    main()