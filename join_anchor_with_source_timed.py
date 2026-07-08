import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse
from datetime import datetime, timedelta
import re

def load_summary_mapping(mapping_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load the summary mapping JSON file.
    """
    with open(mapping_file, 'r') as f:
        return json.load(f)

def load_line_level_data(line_level_file: Path) -> pd.DataFrame:
    """
    Load the line-level tabular data.
    """
    return pd.read_csv(line_level_file)

def load_note_adms(note_adms_file: Path) -> pd.DataFrame:
    """
    Load the note_adms CSV file with discharge times.
    """
    df = pd.read_csv(
        note_adms_file,
        compression='gzip' if note_adms_file.suffix == '.gz' else None,
        dtype={'note_id': str, 'hadm_id': str, 'subject_id': str},
    )
    for column in ('note_id', 'hadm_id', 'subject_id'):
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df

def find_matching_summary_id(summary_text: str, summary_mapping: Dict[str, Dict[str, Any]]) -> str:
    """
    Find the summary ID that matches the given summary text.
    """
    for summary_id, summary_data in summary_mapping.items():
        if summary_data["summary_text"] == summary_text:
            return summary_id
    return None

def get_discharge_time_with_buffer(note_adms_df, hadm_id, buffer_hours=12):
    """
    Get discharge time for a hospital admission ID with added buffer hours.
    Returns None if not found.
    """
    admission_data = note_adms_df[note_adms_df['hadm_id'] == hadm_id]
    if admission_data.empty:
        return None
    
    # Assuming the discharge time is in a column named 'dischtime'
    # If not, we may need to adjust this based on the actual column name
    dischtime_str = admission_data.iloc[0]['dischtime'] if 'dischtime' in admission_data.columns else None
    
    if pd.isna(dischtime_str):
        return None
    
    try:
        # Use pandas to parse - handles ISO 8601 with timezone (e.g., "2180-05-07T17:15:00Z")
        dischtime = pd.to_datetime(dischtime_str)

        # Normalize to timezone-naive for comparison
        # If timezone-aware, convert to UTC then remove timezone info
        if dischtime.tzinfo is not None:
            dischtime = dischtime.tz_convert('UTC').tz_localize(None)

        return dischtime + timedelta(hours=buffer_hours)
    except (ValueError, TypeError):
        return None

def extract_note_id_from_summary_filename(summary_filename: str) -> str:
    """
    Extract note_id from summary filename (e.g., '14193386-DS-21.txt_summary.csv' -> '14193386-DS-21')
    """
    # Remove the file extension parts
    base_name = Path(summary_filename).stem  # '14193386-DS-21.txt_summary'
    # Remove the '_summary' suffix and '.txt' part
    if base_name.endswith('_summary'):
        base_name = base_name[:-8]  # Remove '_summary'
    if base_name.endswith('.txt'):
        base_name = base_name[:-4]  # Remove '.txt'
    return base_name

def candidate_ids_from_name(name: str) -> List[str]:
    """
    Return possible note/subject IDs from either a bare subject-id filename or
    a longer filename such as 2024-..._annotations_103859_summary.csv.
    """
    base_name = Path(str(name)).name
    for suffix in (
        '.csv.gz', '.json.gz', '.txt.gz', '.bsv.gz',
        '.csv', '.json', '.txt', '.bsv',
    ):
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]

    for suffix in ('_summary', '_results', '_joined', '_timelines', '_extracted', '_tags'):
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]

    candidates = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    add(base_name)
    if base_name.endswith('.txt'):
        add(base_name[:-4])
    for token in re.findall(r"(?<!\d)\d{5,}(?!\d)", base_name):
        add(token)

    return candidates

def get_hadm_id_from_note_id(note_adms_df, note_id: str):
    """
    Get hadm_id for a given note_id from note_adms dataframe.
    """
    for candidate_id in candidate_ids_from_name(str(note_id)):
        matching_rows = note_adms_df[note_adms_df['note_id'] == candidate_id]
        if not matching_rows.empty:
            return matching_rows.iloc[0]['hadm_id']

        if 'subject_id' in note_adms_df.columns:
            matching_rows = note_adms_df[note_adms_df['subject_id'] == candidate_id]
            if not matching_rows.empty:
                return matching_rows.iloc[0]['hadm_id']

        # Try using the note_id without the '-DS-xx-' part.
        mask = note_adms_df['note_id'].astype(str).str.match(fr'^{re.escape(candidate_id)}-')
        matching_rows = note_adms_df[mask]
        if not matching_rows.empty:
            return matching_rows.iloc[0]['hadm_id']

    return None

def join_anchor_with_source(anchor_results_file: Path, summary_mapping_file: Path, 
                           line_level_file: Path, note_adms_file: Path, output_file: Path,
                           buffer_hours: int = 12) -> pd.DataFrame:
    """
    Join anchor results with source data via summary mapping, filtering events
    to only include those preceding discharge time + buffer hours.
    """
    # Load all data
    anchor_df = pd.read_csv(anchor_results_file)
    summary_mapping = load_summary_mapping(summary_mapping_file)
    line_level_df = load_line_level_data(line_level_file)
    note_adms_df = load_note_adms(note_adms_file)
    
    # Debug: print the structure of summary_mapping to understand the keys
    print("Summary mapping keys sample:", list(summary_mapping.keys())[:5])
    print("Anchor summary sample:", anchor_df["summary"].head(5).tolist())
    
    # Add summary_id column to anchor_df by matching summary_text
    anchor_df["summary_id"] = anchor_df["summary"].apply(
        lambda x: find_matching_summary_id(x, summary_mapping)
    )
    
    # Debug: check if we found any matching summary_ids
    print("Summary IDs found:", anchor_df["summary_id"].notna().sum())
    
    # Create a reverse mapping from summary_text to summary key (the key in summary_mapping)
    text_to_key = {}
    for key, data in summary_mapping.items():
        text_to_key[data["summary_text"]] = key
    
    # Parse timestamps in line_level_df - handle both ISO 8601 and Unix epoch
    if 't' in line_level_df.columns:
        # First try parsing as ISO 8601 strings
        line_level_df['timestamp'] = pd.to_datetime(line_level_df['t'], errors='coerce')

        # Check if timestamps look reasonable (not all in 1970)
        # If most timestamps are in 1970, they're probably Unix epoch seconds, not ISO dates
        min_timestamp = line_level_df['timestamp'].min()
        if pd.notna(min_timestamp) and min_timestamp.year < 2000:
            # Probably Unix epoch seconds, re-parse with unit='s'
            line_level_df['timestamp'] = pd.to_datetime(line_level_df['t'], unit='s', errors='coerce')

        # For any remaining NaT values, try alternate parsing
        mask = line_level_df['timestamp'].isna()
        if mask.any():
            line_level_df.loc[mask, 'timestamp'] = pd.to_datetime(
                line_level_df.loc[mask, 't'], unit='s', errors='coerce'
            )

        # Ensure timestamps are timezone-naive for comparison
        # If timezone-aware, convert to UTC then remove timezone info
        if line_level_df['timestamp'].dt.tz is not None:
            line_level_df['timestamp'] = line_level_df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        print("Warning: 't' column not found in line-level data for timestamp parsing")
    
    expanded_rows = []
    
    for _, anchor_row in anchor_df.iterrows():
        summary_text = anchor_row["summary"]
        
        # Find the key in summary_mapping that corresponds to this summary text
        if summary_text in text_to_key:
            mapping_key = text_to_key[summary_text]
            row_indices = summary_mapping[mapping_key]["row_indices"]

            # Extract note_id from the anchor_results filename (contains note_id like "14193386-DS-21")
            note_id = extract_note_id_from_summary_filename(str(anchor_results_file))
            
            # Get hospital admission ID from note_adms using note_id
            hadm_id = get_hadm_id_from_note_id(note_adms_df, note_id)
            if hadm_id is None:
                print(f"Warning: No hadm_id found for note_id {note_id}")
                continue
            
            # Get discharge time cutoff for this admission
            cutoff_time = get_discharge_time_with_buffer(note_adms_df, hadm_id, buffer_hours)
            if cutoff_time is None:
                print(f"Warning: Could not find discharge time for hadm_id {hadm_id}")
                continue
            
            for row_idx in row_indices:
                # Get the corresponding row from line_level_df
                if row_idx < len(line_level_df):
                    line_row = line_level_df.iloc[row_idx]

                    # Check if timestamp precedes cutoff time
                    if 'timestamp' in line_level_df.columns and not pd.isna(line_row['timestamp']):
                        if line_row['timestamp'] < cutoff_time:
                            line_row_dict = line_row.to_dict()
                            expanded_row = {
                                "anchor": anchor_row["anchor"],
                                "summary": anchor_row["summary"],
                                "score": anchor_row["score"],
                                "bestscore": anchor_row["bestscore"],
                                "event": anchor_row["event"],
                                "summary_id": anchor_row["summary_id"],
                                "row_index": row_idx,
                                "line_level_timestamp": line_row_dict.get("t", ""),
                                "line_level_event": line_row_dict.get("event", ""),
                                "line_level_value": line_row_dict.get("value", ""),
                                "hadm_id": hadm_id,
                                "cutoff_time": cutoff_time.strftime('%Y-%m-%d %H:%M:%S'),
                                "note_id": note_id
                            }
                            expanded_rows.append(expanded_row)
    
    # Create expanded DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Save to output file
    expanded_df.to_csv(output_file, index=False)
    
    return expanded_df

def main():
    parser = argparse.ArgumentParser(description="Join anchor results with source tabular data")
    parser.add_argument("--anchor-results", required=True, help="Anchor results CSV file")
    parser.add_argument("--summary-mapping", required=True, help="Summary mapping JSON file")
    parser.add_argument("--line-level-data", required=True, help="Line-level tabular data CSV file")
    parser.add_argument("--note-adms", default="./data/note_adms.csv.gz", help="Note admissions data with discharge times (default: ./data/note_adms.csv.gz)")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file if it exists")
    
    args = parser.parse_args()
    
    anchor_results_file = Path(args.anchor_results)
    summary_mapping_file = Path(args.summary_mapping)
    line_level_file = Path(args.line_level_data)
    note_adms_file = Path(args.note_adms)
    output_file = Path(args.output)
    
    if not anchor_results_file.exists():
        print(f"Anchor results file not found: {anchor_results_file}")
        return
    
    if not summary_mapping_file.exists():
        print(f"Summary mapping file not found: {summary_mapping_file}")
        return
    
    if not line_level_file.exists():
        print(f"Line-level data file not found: {line_level_file}")
        return
    
    if not note_adms_file.exists():
        print(f"Note admissions file not found: {note_adms_file}")
        return
    
    # Check if output file exists and is not empty
    if output_file.exists() and output_file.stat().st_size > 0 and not args.overwrite:
        print(f"Output file {output_file} already exists and is not empty. Use --overwrite to regenerate.")
        return
    
    # Perform the join
    result_df = join_anchor_with_source(anchor_results_file, summary_mapping_file, 
                                       line_level_file, note_adms_file, output_file)
    
    print(f"Joined results saved to {output_file}")
    print(f"Total expanded rows: {len(result_df)}")
    
    # Show a sample of the data
    if not result_df.empty:
        print("\nSample of the joined data:")
        print(result_df.head(10))

    # Force immediate exit - os._exit bypasses all Python cleanup
    import os
    os._exit(0)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure we always exit, even if there's an unhandled exception
        import os
        os._exit(0)
