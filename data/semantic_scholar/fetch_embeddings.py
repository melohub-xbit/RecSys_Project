import requests
import json
import time
import os

# Files
INPUT_FILE = os.path.join(os.path.dirname(__file__), "recsys_core_dataset_unauth.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "recsys_enriched_unauth.jsonl")

# Semantic Scholar Batch Endpoint
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
BATCH_SIZE = 50  # Keep this low so the heavy embedding arrays don't break the response

def load_target_ids():
    """Reads the downloaded dataset to get all paper IDs."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the first script to get the core dataset.")
        return []
        
    ids = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line)
            if paper.get("paperId"):
                ids.append(paper["paperId"])
    return ids

def load_processed_ids():
    """Checks the output file to see what we've already downloaded (for resuming)."""
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line)
                if paper.get("paperId"):
                    processed.add(paper["paperId"])
    return processed

def fetch_missing_fields():
    all_ids = load_target_ids()
    processed_ids = load_processed_ids()
    
    # Filter out IDs we've already fetched
    remaining_ids = [pid for pid in all_ids if pid not in processed_ids]
    
    print(f"Total papers to process: {len(all_ids)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining to fetch: {len(remaining_ids)}")
    
    if not remaining_ids:
        print("Everything is fully downloaded!")
        return

    params = {
        "fields": "paperId,tldr,embedding"
    }

    # Open in append mode so we can resume if the script stops
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        
        # Loop through remaining IDs in batches
        for i in range(0, len(remaining_ids), BATCH_SIZE):
            batch_ids = remaining_ids[i : i + BATCH_SIZE]
            payload = {"ids": batch_ids}
            
            while True:
                response = requests.post(BATCH_URL, params=params, json=payload)
                
                if response.status_code == 429:
                    print("Hit rate limit. Sleeping for 60 seconds...")
                    time.sleep(60)
                    continue
                elif response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                    # If a specific batch completely fails, skip it after printing error
                    break 
                
                # Success
                data = response.json()
                for paper in data:
                    # The API might return None for papers it can't find embeddings for
                    if paper is not None: 
                        f.write(json.dumps(paper) + "\n")
                
                print(f"Successfully fetched batch {i // BATCH_SIZE + 1}...")
                
                # Sleep to respect unauthenticated global limits
                print("Sleeping 10 seconds before next batch...")
                time.sleep(10)
                break # Break the retry loop and go to the next batch

if __name__ == "__main__":
    fetch_missing_fields()