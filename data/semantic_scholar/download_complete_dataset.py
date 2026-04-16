import requests
import json
import time
import os

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "recsys_complete_dataset.jsonl")

# Semantic Scholar API Endpoints
BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

BATCH_SIZE = 50  # Keep batch size low for embeddings

# Bulk search parameters
bulk_params = {
    "venue": "RecSys,CIKM,SIGIR,KDD,WWW,WSDM",
    "year": "2018-2026",
    "fields": "paperId,title,year,abstract",
    "limit": 1000 
}

# Batch search parameters
batch_params = {
    "fields": "paperId,tldr,embedding"
}

def load_processed_ids():
    """Checks the output file to see what we've already downloaded (for resuming)."""
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    paper = json.loads(line)
                    if paper.get("paperId"):
                        processed.add(paper["paperId"])
                except Exception:
                    pass
    return processed

def download_complete_dataset():
    processed_ids = load_processed_ids()
    print(f"Loaded {len(processed_ids)} already processed papers.")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        # Loop over bulk search pages
        while True:
            print("Fetching bulk search page...")
            try:
                bulk_resp = requests.get(BULK_URL, params=bulk_params)
            except requests.exceptions.RequestException as e:
                print(f"Network error during bulk search: {e}. Sleeping 10s...")
                time.sleep(10)
                continue

            if bulk_resp.status_code == 429:
                print("Hit rate limit on bulk search. Sleeping for 60 seconds...")
                time.sleep(60)
                continue
            elif bulk_resp.status_code != 200:
                print(f"Error {bulk_resp.status_code} on bulk search: {bulk_resp.text}")
                break
                
            bulk_data = bulk_resp.json()
            papers = bulk_data.get("data", [])
            
            # Map paperId to paper metadata for O(1) lookup
            papers_dict = {
                p["paperId"]: p for p in papers 
                if p.get("paperId") and p["paperId"] not in processed_ids
            }
            
            paper_ids_to_process = list(papers_dict.keys())
            
            if paper_ids_to_process:
                print(f"Found {len(paper_ids_to_process)} new papers in this bulk page. Fetching embeddings...")
                
                # Process in small batches to get tldr and embedding
                for i in range(0, len(paper_ids_to_process), BATCH_SIZE):
                    batch_ids = paper_ids_to_process[i : i + BATCH_SIZE]
                    payload = {"ids": batch_ids}
                    
                    while True:
                        try:
                            batch_resp = requests.post(BATCH_URL, params=batch_params, json=payload)
                        except requests.exceptions.RequestException as e:
                            print(f"Network error during batch search: {e}. Sleeping 10s...")
                            time.sleep(10)
                            continue

                        if batch_resp.status_code == 429:
                            print("Hit rate limit on batch search. Sleeping for 60 seconds...")
                            time.sleep(60)
                            continue
                        elif batch_resp.status_code != 200:
                            print(f"Error {batch_resp.status_code} on batch search: {batch_resp.text}")
                            break # Skip this batch
                            
                        batch_data = batch_resp.json()
                        
                        # Merge batch data into the original paper metadata
                        for enriched_paper in batch_data:
                            if enriched_paper is None or not enriched_paper.get("paperId"):
                                continue
                                
                            pid = enriched_paper["paperId"]
                            if pid in papers_dict:
                                merged_paper = papers_dict[pid].copy()
                                merged_paper["tldr"] = enriched_paper.get("tldr")
                                merged_paper["embedding"] = enriched_paper.get("embedding")
                                
                                # Write immediately completely enriched record
                                f.write(json.dumps(merged_paper) + "\n")
                                processed_ids.add(pid)
                                
                        print(f"Successfully fetched enriched batch {i // BATCH_SIZE + 1} of {-(-len(paper_ids_to_process) // BATCH_SIZE)}...")
                        time.sleep(2) # Modest sleep between batch calls
                        break # Break retry loop

            else:
                print("No new papers in this bulk page (all already processed).")

            # Check if there are more bulk pages
            token = bulk_data.get("token")
            if not token:
                print("Download complete! No more pages.")
                break
                
            bulk_params["token"] = token
            
            # Massive base delay to avoid angering the shared unauth pool for bulk search
            print("Completed bulk page. Waiting 10 seconds before next bulk page...")
            time.sleep(10)

if __name__ == "__main__":
    download_complete_dataset()
