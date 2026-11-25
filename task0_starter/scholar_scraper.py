import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import time

def scrape_arxiv(queries, max_results_per_query=10, output_csv="arxiv_papers.csv"):
    """
    Scrapes the official arXiv API which has no rate limit blocks and requires no authentication.
    """
    papers = []
    seen_ids = set()
    
    for query in queries:
        print(f"\n--- Starting search for: '{query}' ---")
        
        # arXiv API format requires URL encoding. We replace spaces with '+AND+'
        # to ensure it searches for all words.
        search_query = urllib.parse.quote(query.replace(" ", "+AND+"))
        url = f'http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results={max_results_per_query}'
        
        try:
            # Open URL and read XML data
            response = urllib.request.urlopen(url)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            # XML namespace used by arXiv
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', ns)
            
            if not entries:
                print(f"No results found for '{query}'.")
                continue
                
            extracted_count = 0
            for entry in entries:
                paper_id = entry.find('atom:id', ns).text
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                
                title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
                summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
                published = entry.find('atom:published', ns).text[:4] # get year only
                
                authors = []
                for author in entry.findall('atom:author', ns):
                    authors.append(author.find('atom:name', ns).text)
                authors_str = ", ".join(authors)
                
                paper_info = {
                    "Query": query,
                    "Title": title,
                    "Authors": authors_str,
                    "Year": published,
                    "Venue": "arXiv",
                    "Abstract": summary,
                    "Citation Count": "N/A",  # arXiv API doesn't provide citations
                    "URL/DOI": paper_id
                }
                
                papers.append(paper_info)
                extracted_count += 1
                print(f"[{extracted_count}/{max_results_per_query}] Extracted: {title[:60]}...")
                
            # Sleep 3 seconds (as requested by arXiv API documentation)
            time.sleep(3)
            
        except Exception as e:
            print(f"Error fetching data for query '{query}': {e}")
            
    if papers:
        df = pd.DataFrame(papers)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\nSuccessfully saved {len(papers)} unique papers to '{output_csv}'.")
    else:
        print("\nNo papers were extracted.")

if __name__ == "__main__":
    QUERIES = [
        "medical image anomaly detection vision language",
        "CLIP medical anomaly detection",
        "zero-shot anomaly detection medical images",
        "vision-language models medical anomaly detection"
    ]
    MAX_PAPERS_PER_QUERY = 50
    OUTPUT_FILE = "llm_healthcare_papers.csv"
    
    scrape_arxiv(QUERIES, max_results_per_query=MAX_PAPERS_PER_QUERY, output_csv=OUTPUT_FILE)