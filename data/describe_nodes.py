from pathlib import Path

def load_openai():
    from openai import OpenAI
    from dotenv import load_dotenv
    import os

    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    org = os.getenv("OPENAI_ORG_ID")
    client = OpenAI(api_key=key,organization=org)
    return client

def count_tokens(text,model="gpt-4o"):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)   
    return len(encoding.encode(text))

def truncate_pdf(pdf_text,max_tokens=28000):
    pdf_tokens = count_tokens(pdf_text)

    if pdf_tokens <= max_tokens:
        return pdf_text
    return pdf_text[:int(0.92*len(pdf_text))]
    

def create_prompt(pdf, json_f):
    import fitz
    import json
    
    base_prompt_start = """You are an expert at analyzing Bayesian Networks and interpreting scientific documents. Your task is to create clear, contextual descriptions of Bayesian Network nodes based on provided PDF content.

Based on the following PDF content and Bayesian Network nodes information, create a Python dictionary that explains each node in one sentence and the values it can assume. The explanation should be based on the context provided in the PDF and your expert knowledge.

EXAMPLE:
Given nodes for a water quality Bayesian Network, here's the expected format:

{
"C": "C represents conductivity in the water and it can assume values '0' for low conductivity and '1' for high conductivity.",
"Chl_a": "Chl_a stands for chlorophyll-a and is used as an indicator of algal activity, it can be '0' for normal levels and '1' for high concentrations above the threshold.",
"DO": "DO represents dissolved oxygen levels in the water, with '0' indicating low oxygen levels and '1' indicating high oxygen levels.",
"N": "N refers to the concentration of nitrogen in the water, where '0' denotes low nitrogen levels and '1' denotes high nitrogen levels.",
"P": "P symbolizes the concentration of phosphorous in the water, with '0' representing low levels and '1' representing high levels.",
"pH": "pH measures the acidity or alkalinity of the water, with '0' indicating low pH (more acidic) and '1' indicating high pH (more alkaline).",
"Te": "Te denotes the temperature of the water, with '0' indicating lower temperatures and '1' indicating higher temperatures.",
"Tu": "Tu represents turbidity, or the clarity of the water, with '0' for low turbidity (clearer water) and '1' for high turbidity (murkier water)."
}

Each description should:
1. Clearly state what the variable represents in the domain context
2. Explain what each possible value means (e.g., '0' for low, '1' for high)
3. Use domain-appropriate terminology from the PDF content
4. Be concise but informative (one sentence per node)

Now, analyze the following content:"""

    base_prompt_end = """

INSTRUCTIONS:
Please provide a Python dictionary where:
- Keys are the exact node names from the JSON structure
- Values are single-sentence descriptions explaining what each node represents and what its possible values mean
- Descriptions should be grounded in the PDF content and use appropriate domain terminology
- Follow the format shown in the example above
- Ensure all node names from the JSON are included in your response

Return only the Python dictionary in proper format, ready to be parsed."""


    pdf_text = ""
    doc = fitz.open(pdf)
    for page in doc:
        pdf_text += page.get_text()
    doc.close()
    
    pdf_content = "\n\nPDF CONTENT:\n" + pdf_text
    pdf_content = truncate_pdf(pdf_content)
    
    with open(json_f, 'r') as f:
        json_structure = json.load(f)
    
    json_content = "\n\nBAYESIAN NETWORK NODES AND THEIR POSSIBLE STATES:\n" + json.dumps(json_structure, indent=2)
    return base_prompt_start, pdf_content, json_content, base_prompt_end


def model_describe(client,prompt):
    full_prompt = "".join(prompt)
    messages = [
        {"role": "system", "content": "You are an expert at analyzing Bayesian Networks and scientific documents. Provide accurate, contextual descriptions based on the given content."},
        {"role": "user", "content": full_prompt}
    ]

    response = client.chat.completions.create(
            model="gpt-4o-2024-11-20", 
            messages=messages,
            max_tokens=2000,
            temperature=0.3,  # lower temperature for more consistent output
        )
    return response.choices[0].message.content

def log_response(response,name,output_dir='./descriptions'):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True) 
    output_file = output_path / f"{name}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response)
    print(f"Saved response to: {output_file}")

def main():

    client = load_openai()

    pdf_path = Path('./pdfs')
    json_path = Path('./jsons/')
    pdf_files = sorted(pdf_path.iterdir())
    json_files = sorted(json_path.iterdir())
    print(f"Total PDFs: {len(pdf_files)}, Total JSONs: {len(json_files)}")

    exit()

    for pdf_file, json_file in zip(pdf_files, json_files):
        prompt = create_prompt(pdf_file,json_file)
        result = model_describe(client,prompt)
        log_response(result,pdf_file.stem)


if __name__ == '__main__':
    main()