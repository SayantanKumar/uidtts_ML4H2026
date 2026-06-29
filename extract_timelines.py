import argparse
import json
import gzip
import warnings
from pathlib import Path
from typing import List, Dict, Any

def parse_timeline_entry(entry: Dict[str, Any]) -> str:
    """Parse a timeline entry into BSV format."""
    try:
        uid4 = entry.get("uid4", "")
        mention = entry.get("mention", "")
        time = entry.get("time", "")
        bounds = entry.get("bounds", [])
        known = entry.get("known", "")
        context_uid4s = entry.get("context_uid4s", [])
        
        # Format bounds as [lower,upper]
        bounds_str = f"[{bounds[0]},{bounds[1]}]" if len(bounds) == 2 else "[]"
        
        # Format context_uid4s as pipe-separated list
        context_str = "[" + (", ".join(context_uid4s) if context_uid4s else "") + "]"
        
        # Create BSV line
        return f"{uid4}|{mention}|{time}|{bounds_str}|{known}|{context_str}"
    except Exception as e:
        warnings.warn(f"Error parsing entry {entry.get('uid4', 'unknown')}: {str(e)}")
        return None

def has_nonempty_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

def is_bsv_table(text: str) -> bool:
    """Return True when text looks like a non-trivial pipe-separated timeline table."""
    if not isinstance(text, str):
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if "\t" in lines[0]:
        return False
    columns = [col for col in lines[0].split("|") if col.strip()]
    return len(columns) >= 2

def process_timeline_json(json_data: Dict[str, Any], file_name: str) -> List[str]:
    """Process a timeline JSON file and convert to BSV format."""
    bsv_lines = []
    
    try:
        # Get timelines from JSON data
        timelines = json_data.get("timelines", [])
        
        # Process each timeline
        for timeline_idx, timeline_data in enumerate(timelines):
            # Find timeline key (timeline_1, timeline_2, etc.)
            # timeline_key = next((key for key in timeline_data.keys() if key.startswith("timeline_")), None)
            timeline_key = next((key for key in timeline_data.keys()), None)
            
            if not timeline_key:
                warnings.warn(f"No timeline found in timeline {timeline_idx}")
                continue
                
            timeline_entries = timeline_data['data']
            
            # import pdb 
            # pdb.set_trace()
            # Process each entry in the timeline
            for entry in timeline_entries:
                if isinstance(entry, dict):
                    # Handle dictionary format entries
                    bsv_line = parse_timeline_entry(entry)
                    if bsv_line:
                        bsv_lines.append(bsv_line)
                elif isinstance(entry, str):
                    # Handle string format entries
                    parts = entry.split("|")
                    if len(parts) >= 6:
                        # Already in BSV format - use as-is
                        bsv_lines.append(entry)
                    else:
                        warnings.warn(f"Invalid string entry format: {entry}")
                        
    except Exception as e:
        warnings.warn(f"Error processing timeline data: {str(e)}")

def convert_json_to_bsv(json_file_path: str, output_dir: str, verbose: bool = False) -> None:
    """Convert a single JSON timeline file to BSV format."""
    try:
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Get base filename without extension
        base_name = Path(json_file_path).stem
        
        # Process timelines
        timelines = json_data.get("timelines", [])

        for timeline_idx, timeline_data in enumerate(timelines):
            # Find timeline key for each timeline
            # timeline_key = next((key for key in timeline_data.keys() if key.startswith("timeline_")), None)
            timeline_key = next((key for key in timeline_data.keys()), None)
            
            if timeline_key:
                # Generate output filename in output directory
                output_filename = Path(output_dir) / f"{base_name}_timeline_{timeline_idx + 1}.bsv.gz"

                if has_nonempty_file(output_filename):
                    if verbose:
                        print(f"Skipping existing: {output_filename}")
                    continue
                
                # Get timeline entries
                timeline_entries = timeline_data['data']

                try:
                    _ = json.loads(timeline_entries) #is a timeline
                    pass
                except:
                    # is a bsv already
                    if not is_bsv_table(timeline_entries):
                        warnings.warn(
                            f"Skipping invalid timeline data in {json_file_path}; "
                            "expected a pipe-separated BSV table in timeline.data"
                        )
                        continue
                    if verbose:
                        print(f'Already contains a bsv; saving to {output_filename}')
                    with gzip.open(output_filename, 'wt', encoding='utf-8') as f:
                        f.write(timeline_entries.strip()+'\n')
                    continue
                
                # import pdb
                # pdb.set_trace() 
                # Convert entries to BSV
                timeline_bsv = []
                for entry in timeline_entries:
                    if isinstance(entry, dict):
                        # Convert dict entries to BSV
                        bsv_line = parse_timeline_entry(entry)
                        if bsv_line:
                            timeline_bsv.append(bsv_line)
                    elif isinstance(entry, str):
                        # Preserve existing BSV string entries
                        timeline_bsv.append(entry)

                if not timeline_bsv:
                    warnings.warn(f"Skipping empty timeline data in {json_file_path}")
                    continue
                
                # Write compressed file
                with gzip.open(output_filename, 'wt', encoding='utf-8') as f:
                    for line in timeline_bsv:
                        f.write(line + '\n')
                
                if verbose:
                    print(f"Created: {output_filename}")
                
    except json.JSONDecodeError as e:
        warnings.warn(f"Invalid JSON in file {json_file_path}: {str(e)}")
    except Exception as e:
        warnings.warn(f"Error processing file {json_file_path}: {str(e)}")

def batch_convert_json_files(directory: str, output_dir: str, pattern: str = "*.json", verbose: bool = False) -> None:
    """Convert JSON timeline files matching pattern in directory."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find files matching pattern in directory
    json_files = Path(directory).glob(pattern)
    
    for json_file in json_files:
        try:
            if verbose:
                print(f"Processing: {json_file}")
            convert_json_to_bsv(str(json_file), output_dir, verbose)
        except Exception as e:
            warnings.warn(f"Failed to process {json_file}: {str(e)}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Convert timeline JSON files to BSV format')
    parser.add_argument('--directory', '-d', type=str, default=".",
                        help='Directory containing JSON files (default: current directory)')
    parser.add_argument('--pattern', '-p', type=str, default="*.json",
                        help='Wildcard pattern for file selection (default: *.json)')
    parser.add_argument('--output-dir', '-o', type=str, default=".",
                        help='Output directory for BSV files (default: current directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    # Convert files matching the pattern
    # import pdb
    # pdb.set_trace()
    batch_convert_json_files(args.directory, args.output_dir, args.pattern, args.verbose)

    # Explicit exit to prevent hanging
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()
