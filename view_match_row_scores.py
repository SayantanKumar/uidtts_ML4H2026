"""
Parser for anchor_match_ranker.py output files.
Creates a DataFrame with summary, score, and bestscore columns.

Example:
python scripts/parse_mention_results.py --results-dir ./data/e2rr_comparison_results --output ./data/mention_results.csv --sort-by score
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse

def parse_mention_results_file(file_path: Path) -> pd.DataFrame:
    """
    Parse a single mention results JSON file and return a DataFrame.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    
    for event, match_data in data.items():
        uid4 = match_data["uid4"]
        best_score = match_data["best_score"]
        
        # Process reranked results (or initial candidates if no reranking)
        results = match_data.get("reranked_results", match_data.get("initial_candidates", []))
        
        for result in results:
            rows.append({
                "summary": result["content"],
                "score": result["score"],
                "bestscore": best_score,
                "event": event,
                "uid4": uid4
            })
    
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Parse mention results files")
    parser.add_argument("--results-file", required=True, help="Path to the results JSON file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--sort-by", choices=["score", "bestscore"], default="score", 
                       help="Primary sorting column (default: score)")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite existing output file if it exists")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    output_path = Path(args.output)
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    # Check if output file already exists and handle accordingly
    if output_path.exists() and not args.overwrite:
        print(f"Output file already exists: {output_path}")
        print("Use --overwrite to overwrite the existing file")
        return
    
    # Parse the single results file
    df = parse_mention_results_file(results_file)
    
    if df.empty:
        print("No data found in the results file")
        return
    
    # Sort the DataFrame
    if args.sort_by == "score":
        df = df.sort_values(["score", "bestscore"], ascending=[False, False])
    else:
        df = df.sort_values(["bestscore", "score"], ascending=[False, False])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Total rows: {len(df)}")
    
    # Show a sample of the data
    print("\nSample of the data:")
    print(df.head(10))

if __name__ == "__main__":
    main()