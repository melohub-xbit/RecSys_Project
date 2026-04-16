import requests
import json
import time
import os

# --- Configuration ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "s2ag_bigger")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "recsys_complete_dataset_bigger.jsonl")
TOKEN_FILE = os.path.join(OUTPUT_DIR, "resume_token_bigger.txt")

BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

# The expanded parameters you requested
SEARCH_PARAMS = {
    "venue": "RecSys,CIKM,SIGIR,KDD,WWW,WSDM,ECIR,UMAP,ICDM,SDM,NeurIPS,ICML,ICLR,AAAI,IJCAI,ACL,EMNLP,NAACL",
    "year": "2000-2026",
    "fields": "paperId,title,year,abstract",
    "limit": 1000 
}

def get_saved_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            token = f.read().strip()
            if token:
                print(f"Found resume token! Resuming from previous state...")
                return token
    return None

def save_token(token):
    with open(TOKEN_FILE, "w") as f:
        f.write(token)

def safe_request(method, url, **kwargs):
    """Handles the brutal unauthenticated 429 rate limits safely."""
    while True:
        if method == 'GET':
            response = requests.get(url, **kwargs)
        else:
            response = requests.post(url, **kwargs)
            
        if response.status_code == 429:
            print("   [!] 429 Throttled by API. Sleeping 60 seconds...")
            time.sleep(60)
            continue
        elif response.status_code != 200:
            print(f"   [!] Error {response.status_code}: {response.text}")
            return None
            
        return response.json()

def run_pipeline():
    print(f"Starting unauthenticated pipeline. Outputting to {FINAL_OUTPUT_FILE}")
    
    # Load resume token if it exists
    token = get_saved_token()
    if token:
        SEARCH_PARAMS["token"] = token
        
    total_processed = 0

    # Open the final file in append mode
    with open(FINAL_OUTPUT_FILE, "a", encoding="utf-8") as f:
        while True:
            print("\n--- Fetching bulk metadata batch (up to 1000 papers) ---")
            bulk_data = safe_request('GET', BULK_URL, params=SEARCH_PARAMS)
            
            if not bulk_data:
                print("Critical failure fetching bulk data. Exiting.")
                break
                
            papers = bulk_data.get("data", [])
            if not papers:
                print("No more papers found. Pipeline complete!")
                break
                
            print(f"Found {len(papers)} papers. Fetching embeddings in sub-batches...")
            
            # Map paper IDs to their core metadata for easy merging
            paper_dict = {p["paperId"]: p for p in papers if p.get("paperId")}
            paper_ids = list(paper_dict.keys())
            
            # Fetch embeddings & TLDRs in batches of 50 to avoid payload crashes
            batch_size = 50
            for i in range(0, len(paper_ids), batch_size):
                sub_ids = paper_ids[i:i + batch_size]
                
                batch_params = {"fields": "paperId,tldr,embedding"}
                batch_payload = {"ids": sub_ids}
                
                print(f"   -> Enriching chunk {i} to {i+len(sub_ids)}...")
                enriched_data = safe_request('POST', BATCH_URL, params=batch_params, json=batch_payload)
                
                if enriched_data:
                    for enriched_paper in enriched_data:
                        if enriched_paper and enriched_paper.get("paperId"):
                            pid = enriched_paper["paperId"]
                            # Merge the enriched fields into the core dictionary
                            paper_dict[pid]["tldr"] = enriched_paper.get("tldr")
                            paper_dict[pid]["embedding"] = enriched_paper.get("embedding")
                            
                # Sleep to respect unauthenticated limits between batch chunks
                time.sleep(5)

            # Write the fully merged papers to the JSONL file
            for pid, full_paper in paper_dict.items():
                f.write(json.dumps(full_paper) + "\n")
                
            total_processed += len(papers)
            print(f"\n[+] Successfully saved batch. Total papers processed: {total_processed}")
            
            # Check for pagination token
            next_token = bulk_data.get("token")
            if not next_token:
                print("Download complete! No more pagination tokens.")
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE) # Clean up token file
                break
                
            # Save token for next loop and to disk in case of crash
            SEARCH_PARAMS["token"] = next_token
            save_token(next_token)
            
            print("Sleeping 15 seconds before next bulk request...")
            time.sleep(15)

if __name__ == "__main__":
    run_pipeline()