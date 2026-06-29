#!/usr/bin/env python3
"""
Extract tag_number, uid4, and mention from LLM outputs (txt.gz files).
Creates BSV files with format: tag_number|uid4|mention
Works with prompt_io.py outputs directly.
"""

import argparse
import gzip
import re
import os
import uuid
from pathlib import Path
from typing import List, Dict

def generate_unique_uids(num_needed: int, excess: int = 100) -> List[str]:
    """Generate unique 4-character UUIDs with collision handling."""
    if num_needed > 1000000:
        raise ValueError("Cannot generate more than 1 million unique uids")
    
    total_needed = num_needed + excess
    max_attempts = 10
    
    for attempt in range(max_attempts):
        # Generate pool of candidate uids
        candidate_pool = set()
        while len(candidate_pool) < total_needed:
            candidate_pool.add(str(uuid.uuid4())[:4])
        
        # Check if we have enough
        if len(candidate_pool) >= num_needed:
            return sorted(list(candidate_pool))[:num_needed]
        
        # Increase excess for next attempt
        excess += 100
        total_needed = num_needed + excess
        
        # Safety check
        if excess > 1000000:
            break
    
    raise ValueError(f"Unable to generate {num_needed} unique uids after {max_attempts} attempts")

def extract_tagged_section(content: str) -> str:
    """Extract only the tagged text section from LLM output."""
    # Look for ###Tagged Text### section
    tagged_pattern = r'# ###Tagged Text###\n(.*)'
    match = re.search(tagged_pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # If no section header, assume entire content is tagged
        return content.strip()

def extract_tags_from_content(content: str) -> List[Dict[str, str]]:
    """Extract tag information from tagged content."""
    tags = []
    
    # First, extract only the tagged section
    tagged_content = extract_tagged_section(content)
    
    # Pattern to match: <tag_X mention="text">...</tag_X>
    # pattern = r'<tag_(\d+)\s+mention="([^"]*)"[^>]*>(.*?)</tag_\1>' # strict tag_number matching
    pattern = r'<tag_(\d+)\s+mention="([^"]*)"[^>]*>((?:(?!<tag_\d+).)*?)</tag_\d+>' #loose but nesting accounted for
    matches = re.findall(pattern, tagged_content, re.DOTALL)
    
    # Collect all tags first to determine how many we need
    raw_tags = []
    for tag_num_str, mention, inner_text in matches:
        try:
            tag_num = int(tag_num_str)
            raw_tags.append((tag_num, mention.strip()))
        except ValueError:
            continue
    
    # Also try pattern without mention attribute: <tag_X>text</tag_X>
    simple_pattern = r'<tag_(\d+)[^>]*>(.*?)</tag_\1>'
    simple_matches = re.findall(simple_pattern, tagged_content, re.DOTALL)
    
    for tag_num_str, inner_text in simple_matches:
        try:
            tag_num = int(tag_num_str)
            if not any(t[0] == tag_num for t in raw_tags):
                raw_tags.append((tag_num, inner_text.strip()))
        except ValueError:
            continue
    
    # Generate unique uids for this report
    num_tags = len(raw_tags)
    if num_tags > 0:
        unique_uids = generate_unique_uids(num_tags)
    else:
        unique_uids = []
    
    # Create final tags with unique uids
    for i, (tag_num, mention) in enumerate(sorted(raw_tags)):
        tags.append({
            'tag_number': tag_num,
            'uid4': unique_uids[i],
            'mention': mention
        })
    
    return tags

def process_file(input_file: str, output_file: str) -> None:
    """Process a single LLM output file."""
    
    # Read input file
    open_func = gzip.open if input_file.endswith('.gz') else open
    with open_func(input_file, 'rt', encoding='utf-8') as f:
        content = f.read()
    
    # Extract tags
    tags = extract_tags_from_content(content)
    
    # Write output
    if os.path.exists(output_file):
        print(f'Skipping {output_file}; exists')
        return
    output_open_func = gzip.open if output_file.endswith('.gz') else open
    with output_open_func(output_file, 'wt', encoding='utf-8') as f:
        for tag in tags:
            f.write(f"{tag['tag_number']}|{tag['uid4']}|{tag['mention']}\n")

def process_directory(input_dir: str, output_dir: str) -> None:
    """Process all .txt.gz files in directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt.gz') or filename.endswith('.txt'):
            base_name = filename.replace('.txt.gz', '').replace('.txt', '')
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{base_name}_tags.bsv.gz")
            
            try:
                print(f"Processing {filename}...")
                process_file(input_path, output_path)
                print(f"  Created: {os.path.basename(output_path)}")
            except Exception as e:
                print(f"  Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract tag_number, uid4, and mention from LLM output files."
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input file path (single file mode)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory to process all LLM output files (batch mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_tags",
        help="Output directory for extracted tag tables"
    )
    
    args = parser.parse_args()
    
    if not args.input and not args.input_dir:
        parser.error("Must specify either --input or --input-dir")
    
    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input-dir")
    
    if args.input:
        # Single file mode
        input_path = args.input
        base_name = Path(input_path).stem
        output_path = os.path.join(args.output_dir, f"{base_name}_tags.bsv.gz")
        
        os.makedirs(args.output_dir, exist_ok=True)
        process_file(input_path, output_path)
        print(f"Processed single file: {input_path}")
        
    elif args.input_dir:
        # Batch mode
        if not os.path.exists(args.input_dir):
            print(f"Error: Directory {args.input_dir} does not exist")
            return
            
        process_directory(args.input_dir, args.output_dir)
        print(f"Processed all files in: {args.input_dir}")

if __name__ == "__main__":
    main()