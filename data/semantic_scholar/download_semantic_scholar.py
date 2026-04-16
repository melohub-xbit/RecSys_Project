import requests
import json
import time
import os

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "recsys_core_dataset_unauth.jsonl")
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

params = {
    "venue": "RecSys,CIKM,SIGIR,KDD,WWW,WSDM",
    "year": "2018-2026",
    "fields": "paperId,title,year,abstract",
    "limit": 1000 
}

# Notice: We completely removed the headers dictionary

def fetch_corpus_unauth():
    print(f"Starting unauthenticated bulk download to {OUTPUT_FILE}...")
    total_papers = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        while True:
            # No headers passed here
            response = requests.get(BASE_URL, params=params)
            
            if response.status_code == 429:
                print("Hit the global rate limit pool! Sleeping for 60 seconds...")
                # You must sleep for a long time unauth, or you will get IP blocked
                time.sleep(60) 
                continue
                
            response.raise_for_status()
            data = response.json()
            
            papers = data.get("data", [])
            for paper in papers:
                f.write(json.dumps(paper) + "\n")
                
            total_papers += len(papers)
            print(f"Fetched {total_papers} papers so far...")
            
            token = data.get("token")
            if not token:
                print("Download complete!")
                break
                
            params["token"] = token
            
            # Massive base delay to avoid angering the shared unauth pool
            print("Waiting 15 seconds before next page to respect unauth limits...")
            time.sleep(15) 

if __name__ == "__main__":
    fetch_corpus_unauth()