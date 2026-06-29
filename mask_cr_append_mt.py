#!/usr/bin/env python3
"""
Process case reports to replace tag numbers with uid4 codes from BSV mention tables.
"""

import os
import gzip
import re
from pathlib import Path
import pdb

def has_nonempty_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def process_text_file(text_path, table_path, output_dir):
    """Process a single case report and its mention table."""
    text_filename = os.path.basename(text_path).replace('.txt.gz', '.txt')
    text_output_path = os.path.join(output_dir, text_filename)
    table_filename = os.path.basename(table_path).replace('_tags.bsv.gz', '_uid4.bsv')
    table_output_path = os.path.join(output_dir, table_filename)

    if has_nonempty_file(text_output_path) and has_nonempty_file(table_output_path):
        print(f"Skipping {text_filename} and {table_filename}; outputs already exist")
        return text_filename, table_filename
    
    # Read the text file
    with gzip.open(text_path, 'rt', encoding='utf-8') as f:
        text_content = f.read()
    
    # Read the mention table (BSV format with 3 columns: tag_id|uid4|mention)
    tag_to_uid4 = {}
    # pdb.set_trace()
    with gzip.open(table_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('|')
                if len(parts) >= 3:
                    tag_id = parts[0]
                    uid4 = parts[1]
                    mention = parts[2]
                    # Remove 'tag_' prefix if present
                    tag_num = tag_id.replace('tag_', '')
                    tag_to_uid4[tag_num] = uid4
    
    # Replace each tag with its uid4
    for tag_num, uid4 in tag_to_uid4.items():
        pattern = f'<tag_{tag_num}(?:\\s+mention="[^"]*")?>(.*?)</tag_{tag_num}>'
        replacement = f'<{uid4}>\\1</{uid4}>'
        text_content = re.sub(pattern, replacement, text_content)
    
    # Write modified text
    with open(text_output_path, 'w', encoding='utf-8') as f:
        table_str = 'uid4 | Mention \n'
        with gzip.open(table_path, 'rt', encoding='utf-8') as src:
            for line in src:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 3:
                        # Output format: uid4|mention
                        table_str = table_str + f"{parts[1]}|{parts[2]}\n"
        f.write(text_content + "\n\nMention table:\n" + table_str + "\n")
    
    # Create new mention table with uid4|mention format
    with gzip.open(table_path, 'rt', encoding='utf-8') as src:
        with open(table_output_path, 'w', encoding='utf-8') as dst:
            for line in src:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 3:
                        # Output format: uid4|mention
                        dst.write(f"{parts[1]}|{parts[2]}\n")
    
    return text_filename, table_filename

def process_directories(mention_table_dir, tagged_case_report_dir, output_dir):
    """Process all files in the two input directories."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    mention_table_path = Path(mention_table_dir)
    case_report_path = Path(tagged_case_report_dir)
    processed_files = []
    
    # Process all text.gz files in tagged_case_report_dir
    for text_file in case_report_path.glob('*.txt.gz'):
        # Extract base name (before first '.')
        base_name = text_file.name.split('.')[0]
        
        # Find corresponding table file (base_name + "_temporal.lastblock")
        table_file = mention_table_path / f"{base_name}_tags.bsv.gz"
        # pdb.set_trace()
        if table_file.exists():
            print(f"Processing {text_file.name} and {table_file.name}")
            try:
                text_out, table_out = process_text_file(text_file, table_file, output_dir)
                processed_files.append((text_out, table_out))
            except Exception as e:
                print(f"Error processing {text_file.name}: {e}")
        else:
            print(f"Warning: No matching table file found for {text_file.name}")
    
    return processed_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process case reports with uid4 tags from BSV files')
    parser.add_argument('--mention_table_dir', help='Directory containing mention table BSV files')
    parser.add_argument('--tagged_case_report_dir', help='Directory containing tagged case report TXT.GZ files')
    parser.add_argument('--output_dir', help='Directory for output files')
    
    args = parser.parse_args()
    
    processed = process_directories(args.mention_table_dir, args.tagged_case_report_dir, args.output_dir)
    print(f"\nProcessed {len(processed)} file pairs")
    for text_file, table_file in processed:
        print(f"  - {text_file} and {table_file}")
