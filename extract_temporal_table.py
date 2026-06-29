#!/usr/bin/env python3
"""
Extract columns from temporal analysis output.
Creates BSV files with all columns or specified columns only.
Supports both thinking tokens and answer block formats.
"""

import argparse
import gzip
import re
import os
from pathlib import Path

def extract_table_section(content: str) -> str:
    """Extract the table section from answer blocks or after thinking tokens."""
    
    # First, try to find content within <answer> tags
    # Find the position of the last <answer> tag and extract from there to </answer>
    last_answer_start = content.rfind('<answer>')
    if last_answer_start != -1:
        # Find the closing </answer> after the last <answer>
        answer_end = content.find('</answer>', last_answer_start)
        if answer_end != -1:
            # Extract content between last <answer> and its closing </answer>
            answer_content = content[last_answer_start + len('<answer>'):answer_end]
            return answer_content.strip()
    
    # Look for content after the last occurrence of either </think> or </thinking>
    patterns = [
        r'</think>\s*\n(.*)',
        r'</thinking>\s*\n(.*)'
    ]
    
    for pattern in patterns:
        # Find all matches
        matches = list(re.finditer(pattern, content, re.DOTALL))
        if matches:
            # Use the last match
            return matches[-1].group(1).strip()
    
    # If no thinking tokens or answer blocks found, try to find table directly
    # Look for any line with pipe-separated values that looks like a header
    table_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\s*\|.*?)(?=\n\n|\Z)'
    table_match = re.search(table_pattern, content, re.DOTALL | re.IGNORECASE)
    if table_match:
        return table_match.group(1).strip()
    
    return ""

def parse_table_rows(table_content: str, include_header: bool, keep_columns: list = None, no_header: bool = False) -> tuple:
    """Parse table rows and extract columns based on keep_columns list.
    
    Args:
        table_content: The table content to parse
        include_header: Whether to include header in output
        keep_columns: List of column names to keep (or assign if no_header=True)
        no_header: If True, table has no header row and keep_columns assigns names by index
    
    Returns:
        tuple: (column_names, rows) where rows is a list of dicts
    """
    rows = []
    lines = table_content.split('\n')
    column_names = []
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line or '|' not in line:
            continue
        
        # Skip markdown table separator rows (e.g., ---|---|---)
        if re.match(r'^[\s\-\|:]+$', line):
            continue
        
        # Split by pipe and clean up
        parts = [part.strip() for part in line.split('|')]
        parts = [p for p in parts if p]  # Remove empty strings from leading/trailing pipes
        
        if not parts:
            continue
        
        # Handle no_header mode: assign column names from keep_columns or generate defaults
        if no_header and not column_names:
            if keep_columns:
                # Use keep_columns names for the first N columns
                column_names = keep_columns[:len(parts)]
            else:
                # Generate default column names: col0, col1, col2, ...
                column_names = [f'col{i}' for i in range(len(parts))]
            
            # Add header row if requested
            if include_header:
                rows.append({col: col for col in column_names})
        
        # First non-separator row is the header (if not in no_header mode)
        if not no_header and not column_names:
            column_names = parts
            if include_header:
                # If keeping specific columns, filter the header
                if keep_columns:
                    filtered_parts = []
                    for col in keep_columns:
                        if col in column_names:
                            filtered_parts.append(col)
                    if filtered_parts:
                        rows.append({col: col for col in filtered_parts})
                else:
                    rows.append({col: col for col in column_names})
            continue
        
        # Parse data rows
        if len(parts) >= len(column_names):
            row_dict = {}
            for col_name, value in zip(column_names, parts):
                # If keep_columns is specified, only include those columns
                if keep_columns is None or col_name in keep_columns:
                    row_dict[col_name] = value
            
            if row_dict:  # Only add if there's data after filtering
                rows.append(row_dict)
    
    # Determine final column order
    if keep_columns:
        final_columns = [col for col in keep_columns if col in column_names]
    else:
        final_columns = column_names
    
    return final_columns, rows

def process_file(input_file: str, output_file: str, include_header: bool, keep_columns: list = None, no_header: bool = False) -> None:
    """Process a single temporal analysis file."""
    
    # Read input file
    open_func = gzip.open if input_file.endswith('.gz') else open
    with open_func(input_file, 'rt', encoding='utf-8') as f:
        content = f.read()
    
    # Extract table section
    table_content = extract_table_section(content)
    
    if not table_content:
        print(f"Warning: No table content found in {input_file}")
        return
    
    # Parse table rows
    column_names, rows = parse_table_rows(table_content, include_header, keep_columns, no_header)
    
    if not column_names:
        print(f"Warning: No columns found in {input_file}")
        # import pdb
        # pdb.set_trace()
        return
    
    # Write output
    output_open_func = gzip.open if output_file.endswith('.gz') else open
    with output_open_func(output_file, 'wt', encoding='utf-8') as f:
        # Write rows
        for row in rows:
            values = [row.get(col, '') for col in column_names]
            f.write('|'.join(values) + '\n')

def process_directory(input_dir: str, output_dir: str, include_header: bool, keep_columns: list = None, no_header: bool = False) -> None:
    """Process all temporal analysis files in directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('_temporal.txt.gz') or filename.endswith('_temporal.txt'):
            base_name = filename.replace('_temporal.txt.gz', '').replace('_temporal.txt', '')
        else:
            base_name = filename.replace('.txt.gz','').replace('.txt','')
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{base_name}_extracted.bsv.gz")
        
        try:
            print(f"Processing {filename}...")
            process_file(input_path, output_path, include_header, keep_columns, no_header)
            print(f"  Created: {os.path.basename(output_path)}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract columns from temporal analysis output. By default keeps all columns."
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input file path (single file mode)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory to process all temporal analysis files (batch mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_temporal",
        help="Output directory for extracted tables"
    )
    parser.add_argument(
        "--include-header",
        action='store_true',
        default=False,
        help="Include table header in output"
    )
    parser.add_argument(
        "--keep-columns",
        type=str,
        nargs='+',
        default=None,
        help="Specific columns to keep (default: keep all columns). Example: --keep-columns uid4 mention time. "
             "When used with --no-header, assigns these names to columns by index."
    )
    parser.add_argument(
        "--no-header",
        action='store_true',
        default=False,
        help="Table has no header row. If --keep-columns is specified, those names are assigned to columns by index. "
             "Otherwise, columns are named col0, col1, col2, etc."
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
        output_path = os.path.join(args.output_dir, f"{base_name}_extracted.bsv.gz")
        
        os.makedirs(args.output_dir, exist_ok=True)
        process_file(input_path, output_path, args.include_header, args.keep_columns, args.no_header)
        print(f"Processed single file: {input_path}")
        
    elif args.input_dir:
        # Batch mode
        if not os.path.exists(args.input_dir):
            print(f"Error: Directory {args.input_dir} does not exist")
            return
            
        process_directory(args.input_dir, args.output_dir, args.include_header, args.keep_columns, args.no_header)
        print(f"Processed all files in: {args.input_dir}")

if __name__ == "__main__":
    main()