from pathlib import Path
import pandas as pd
from browser_use import Agent,BrowserSession,Controller,ActionResult
from browser_use.llm import ChatGoogle
from dotenv import load_dotenv
from googlesearch import search
import time
import requests
import os
from bs4 import BeautifulSoup

load_dotenv()

def get_search_names(filtered_path):
    names = []
    for f in Path(filtered_path).iterdir():
        names.append(f.stem)

    return names


def parse_quries(names,summary_path):
    from collections import defaultdict

    query_2_name = defaultdict(list)
    summary = pd.read_csv(summary_path)
    # BASE_URL = "https://www.google.co.in/search?q="
    BASE_URL = ''

    for name in names:
        if name == 'crypto':
            query = '2024, Knowledge and Information System, Amirzadeh, R., Thiruvady, D., Nazari, A., & Ee, M. S. Dynamic evolution of causal relationships among cryptocurrencies: an analysis via Bayesian networks.'
            # query = '+'.join(query.split())
        elif 'salmonella' in name:
            query = 'Teng, K. T. Y., Aerts, M., Jaspers, S., Ugarte-Ruiz, M., Moreno, M. A., Saez, J. L., ... & Alvarez, J. (2022). Patterns of antimicrobial resistance in Salmonella isolates from fattening pigs in Spain. BMC Veterinary Research, 18(1), 333'
            # query = '+'.join(query.split())
        else:
            query = str(summary[summary['Name'] == name][['Year','Journal','Reference']].values[0])
            # query = '+'.join([str(q) for q in query.values[0]])
        
        query = query.replace("'","").replace('"',"").replace("&","").replace("[","").replace("]","")
        query = BASE_URL + query
        # print(query,type(query))
        query_2_name[query].append(name)

    return query_2_name

def get_pdf_links(queries):
    print(len(queries))
    name_to_query = {}
    start_processing = False  # flag to begin only after 'earthquake'

    for q, names in queries.items():
        # Check if we should start processing
        print(names)
        if not start_processing:
            if 'ropesegment' in names:
                start_processing = True
            else:
                continue
        try:
            link = next(search(query=q,num=1,stop=1,pause=35.0))
            # link = ' '
            print(link,names)
            for name in names:
                name_to_query[name] = link
        except Exception as e:
            print(f"ERROR: Could not get papers for {names} b/c {e}. Going to sleep...")
            time.sleep(120)
            link = ''
        # print(name_to_query)
        # print()


import requests
from bs4 import BeautifulSoup
import os
import shutil
import pandas as pd
from urllib.parse import urljoin, urlparse

def extract_download_link(page_url):
    """Extract PDF download link from a page URL."""
    try:
        # Check if it's already a direct PDF link
        if page_url.lower().endswith('.pdf'):
            return page_url
            
        # Special handling for known patterns
        if 'arxiv.org/abs/' in page_url:
            # Convert abstract URL to PDF URL
            # https://arxiv.org/abs/2406.05764 -> https://arxiv.org/pdf/2406.05764.pdf
            paper_id = page_url.split('/abs/')[-1]
            return f"https://arxiv.org/pdf/{paper_id}.pdf"
        
        # For other URLs, try to parse the page
        resp = requests.get(page_url, timeout=20)
        
        # Check if the response itself is a PDF
        if resp.headers.get("Content-Type", "").lower().startswith("application/pdf"):
            return page_url
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Strategy 1: Look for PDF links (improved)
        pdf_links = []
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            # Check for .pdf extension or PDF in text
            if '.pdf' in href or 'pdf' in a.get_text().lower():
                full_url = urljoin(page_url, a['href'])
                pdf_links.append((full_url, a))
        
        # Prioritize links with "download" or "pdf" in the text
        for url, tag in pdf_links:
            text = tag.get_text().lower()
            if 'download' in text or 'pdf' in text:
                return url
                
        # Return first PDF link if any found
        if pdf_links:
            return pdf_links[0][0]
            
        # Strategy 2: Look for download buttons/links
        for tag in soup.find_all(['a', 'button']):
            text = tag.get_text().strip().lower()
            if any(word in text for word in ['download', 'pdf', 'full text']):
                if tag.has_attr('href'):
                    return urljoin(page_url, tag['href'])
                # Check for onclick attributes that might contain URLs
                if tag.has_attr('onclick'):
                    onclick = tag['onclick']
                    if '.pdf' in onclick:
                        # Extract URL from onclick (basic pattern matching)
                        import re
                        pdf_match = re.search(r'["\']([^"\']*\.pdf)["\']', onclick)
                        if pdf_match:
                            return urljoin(page_url, pdf_match.group(1))
                            
    except Exception as e:
        print(f"[ERROR] extract_download_link({page_url}): {e}")
    return None

def download_file(pdf_url, save_path, max_retries=3):
    """Download one PDF to save_path with retries; return True on success."""
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            r = requests.get(pdf_url, stream=True, timeout=30, headers=headers)
            
            # Check if it's actually a PDF
            content_type = r.headers.get("Content-Type", "").lower()
            if not (content_type.startswith("application/pdf") or 
                    content_type == "application/octet-stream"):
                # Sometimes PDFs are served as octet-stream
                # Check first few bytes for PDF signature
                first_chunk = r.content[:5]
                if not first_chunk.startswith(b'%PDF'):
                    print(f"[WARN] Not a PDF: {pdf_url} (Content-Type: {content_type})")
                    return False
                    
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify the file is not empty
            if os.path.getsize(save_path) > 0:
                return True
            else:
                os.remove(save_path)
                print(f"[WARN] Empty file downloaded from {pdf_url}")
                
        except Exception as e:
            print(f"[ERROR] download_file({pdf_url}) attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # Brief pause before retry
                
    return False

def process_links_file(filepath, output_dir='./pdfs'):
    """Process links file and download PDFs."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the line
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    print(f"[ERROR] Line {line_num}: Missing names - {line}")
                    continue
                    
                url, names_str = parts
                names = eval(names_str)
                
            except Exception as e:
                print(f"[ERROR] Line {line_num}: {e} - {line}")
                continue
            
            # Extract PDF link
            pdf_link = extract_download_link(url)
            if not pdf_link:
                print(f"[WARN] No PDF link found for: {url}")
            
            # Process each name
            for name in names:
                # Clean up the name
                base = name[:-4] if name.lower().endswith('.pdf') else name
                # Remove any invalid filename characters
                base = "".join(c for c in base if c.isalnum() or c in (' ', '-', '_')).rstrip()
                save_path = os.path.join(output_dir, f"{base}.pdf")
                
                # Handle existing directory
                if os.path.isdir(save_path):
                    print(f"[WARN] Removing existing directory at {save_path}")
                    shutil.rmtree(save_path)
                
                # Download the PDF
                ok = False
                if pdf_link:
                    print(f"Downloading: {base} from {pdf_link}")
                    ok = download_file(pdf_link, save_path)
                    if ok:
                        print(f"✓ Success: {base}")
                    else:
                        print(f"✗ Failed: {base}")
                
                rows.append({
                    'name': name,
                    'article_url': url,
                    'pdf_link': pdf_link,
                    'pdf_path': save_path if ok else None,
                    'success': ok
                })
    
    # Save results
    df = pd.DataFrame(rows)
    df.to_csv('name_to_pdf.csv', index=False)
    
    # Print summary
    total = len(df)
    successful = df['success'].sum()
    print(f"\n=== Summary ===")
    print(f"Total papers: {total}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Results saved to: name_to_pdf.csv")
    
    return df

if __name__ == "__main__":
    df = process_links_file('links')
    print("\nFirst 5 entries:")
    print(df.head())
    
    # Show failed downloads if any
    failed = df[~df['success']]
    if len(failed) > 0:
        print(f"\nFailed downloads ({len(failed)}):")
        for _, row in failed.iterrows():
            print(f"  - {row['name']} from {row['article_url']}")