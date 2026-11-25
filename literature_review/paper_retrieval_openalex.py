"""
Paper Retrieval via OpenAlex API
Replaces paper_retrieval.py (Semantic Scholar) with OpenAlex as the data source.
Produces the same output JSON format for downstream pipeline compatibility.
"""
import requests
import time
import json
import os
import re
import csv
import sys

INPUT_FILE = "./example/topic.csv"
OUTPUT_DIR = "./"
MAX_RESULTS = 300
API_MAILTO = "surveygen.pipeline@gmail.com"

def sanitize_filename(text: str) -> str:
    text = re.sub(r'[\\/*?"<>|:\n\r\t]', '_', text)
    return text.strip()[:100] or "untitled"

def invert_abstract(inverted_index):
    """Reconstruct abstract from OpenAlex inverted index"""
    if not inverted_index:
        return ""
    word_positions = {
        pos: word
        for word, positions in inverted_index.items()
        for pos in positions
    }
    if not word_positions:
        return ""
    return ' '.join([word_positions.get(i, '') for i in range(max(word_positions.keys()) + 1)])

def search_openalex(query_keywords, year_end, max_results=300):
    """Search OpenAlex for papers matching keywords"""
    collected = []
    page = 1
    per_page = 50  # OpenAlex max per page

    # Build search query
    search_query = query_keywords.replace(";", " ").replace(",", " ").strip()

    while len(collected) < max_results:
        url = "https://api.openalex.org/works"
        params = {
            "search": search_query,
            "filter": f"from_publication_date:2000-01-01,to_publication_date:{year_end}-12-31,has_doi:true",
            "sort": "cited_by_count:desc",
            "per_page": per_page,
            "page": page,
            "mailto": API_MAILTO
        }

        print(f"Searching OpenAlex: page {page} | collected: {len(collected)}/{max_results}")

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                print(f"Rate limit hit, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text[:200]}")
                break

            data = response.json()
            results = data.get("results", [])
            if not results:
                print("No more results.")
                break

            for item in results:
                doi = item.get("doi", "")
                if doi:
                    doi = doi.replace("https://doi.org/", "")

                abstract = invert_abstract(item.get("abstract_inverted_index"))
                if not doi or not abstract:
                    continue

                # Extract journal name
                journal_name = ""
                primary_location = item.get("primary_location") or {}
                source = primary_location.get("source") or {}
                journal_name = source.get("display_name", "")

                collected.append({
                    "corpusId": item.get("id", "").replace("https://openalex.org/", ""),
                    "doi": doi,
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "journalName": journal_name
                })

                if len(collected) >= max_results:
                    break

            page += 1
            time.sleep(1)  # Be polite

        except requests.exceptions.Timeout:
            print(f"Timeout on page {page}, retrying...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"Error: {e}")
            break

    return collected[:max_results]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        reader = list(csv.DictReader(infile))

        for i, row in enumerate(reader):
            if i >= 1:
                break

            title = row["title"].strip()
            year = row["year"].strip()
            keywords_raw = row["keywords"].strip()

            if not (title and year and keywords_raw):
                print("Skipping due to missing title/year/keywords.")
                continue

            year = int(year)
            keyword_query = keywords_raw

            print(f"\nRetrieving papers for: {title}")
            print(f"Keywords: {keyword_query}")
            print(f"Year <= {year}")

            result_list = search_openalex(keyword_query, year, MAX_RESULTS)

            if result_list:
                safe_title = sanitize_filename(title)
                filename = f"{safe_title}.json"
                output_path = os.path.join(OUTPUT_DIR, filename)
                with open(output_path, "w", encoding="utf-8") as out_file:
                    json.dump(result_list, out_file, ensure_ascii=False, indent=2)
                print(f"Saved {len(result_list)} papers to: {output_path}")
            else:
                print("No valid results found.")

    print(f"\nDone. Results saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
