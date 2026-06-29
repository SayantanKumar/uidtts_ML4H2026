import os
import re
import argparse
from pathlib import Path
import csv
import gzip

def has_nonempty_file(path):
    return path.exists() and path.stat().st_size > 0

def extract_tag_positions(text):
    """Extract character positions of open and close tags, adjusting for tag length."""
    positions = []
    cumulative_offset = 0  # Track how much we've shifted due to tag removal
    
    # Updated pattern to handle tags with additional attributes
    # Pattern matches: <uid[ attributes...]>content</uid>
    tag_pattern = r'<([0-9a-fA-F]{4})(?:\s[^>]*)?>(.*?)</\1>'
    
    for match in re.finditer(tag_pattern, text):
        uid = match.group(1)
        content = match.group(2)
        
        # Calculate the actual length of the opening tag
        opening_tag_match = re.search(r'<[0-9a-fA-F]{4}(?:\s[^>]*)?>', match.group(0))
        opening_tag = opening_tag_match.group(0)
        opening_tag_length = len(opening_tag)
        
        # Calculate positions in the *original* text
        open_tag_start = match.start()
        open_tag_end = open_tag_start + opening_tag_length
        
        # Find the closing tag position within the match
        closing_tag_pattern = f'</{uid}>'
        close_tag_start = match.start() + match.group(0).rfind(closing_tag_pattern)
        close_tag_end = close_tag_start + len(closing_tag_pattern)
        
        # Adjust positions to exclude the tag characters, accounting for cumulative offset
        content_start = open_tag_end - cumulative_offset
        content_end = close_tag_start - cumulative_offset
        
        positions.append({
            'uid': uid,
            'char_pos': content_start,
            'char_pos_ub': content_end
        })
        
        # Update cumulative offset for next match
        # Total tag length to remove = opening tag + closing tag
        total_tag_length = opening_tag_length + len(closing_tag_pattern)
        cumulative_offset += total_tag_length
    
    return positions

def load_mentions(mention_file):
    """Load mentions from a two-column BSV file (uid4, mention)."""
    mentions = {}
    with open(mention_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) >= 2:
                mentions[row[0]] = row[1]
    return mentions

def load_timeline(timeline_file):
    """Load timeline data from a BSV file."""
    timeline_data = []
    with gzip.open(timeline_file, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) >= 6:
                timeline_data.append({
                    'uid4': row[0],
                    'mention': row[1],
                    'time': row[2],
                    'bounds': row[3],
                    'known': row[4],
                    'context_uid4s': row[5]
                })
    return timeline_data

def process_file(input_file, output_file, join_mention=False, mention_file=None, timeline_dir=None, output_path=None):
    """Process a single file and write positions to output."""
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    positions = extract_tag_positions(text)
    
    # Load mentions if join_mention is enabled
    mentions = {}
    if join_mention and mention_file and mention_file.exists():
        mentions = load_mentions(mention_file)
    
    # Load timeline data if timeline_dir is provided
    timeline_data = []
    if timeline_dir:
        base_name = input_file.stem.replace('.txt', '')
        timeline_files = list(timeline_dir.glob(f"{base_name}_timelines_timeline_*.bsv.gz"))
        timeline_files += list(timeline_dir.glob(f"{base_name}_extracted.bsv.gz"))
        
        for timeline_file in timeline_files:
            timeline_index = re.sub(r"(\.[^.]+)+$", "", timeline_file.name).split('_')[-1]  # Extract index from filename
            if timeline_index == '':
                timeline_index = '1'
            # print(f'ti: {timeline_index}')
            timeline_data = load_timeline(timeline_file)
            
            # Create output file for this timeline
            timeline_output_file = output_path / f"{base_name}_positions_timeline_{timeline_index}.bsv"

            renamed_unrevised_file = None
            if timeline_index == 'extracted':
                renamed_unrevised_file = output_path / f"{base_name}_positions_1.bsv"

            if has_nonempty_file(timeline_output_file) or (
                renamed_unrevised_file is not None and has_nonempty_file(renamed_unrevised_file)
            ):
                print(f"Skipping existing charpos output for {base_name}, timeline {timeline_index}")
                continue
            
            with open(timeline_output_file, 'w', encoding='utf-8') as f:
                f.write("uid4|char_pos|char_pos_ub|mention|time|bounds|known|context_uid4s\n")
                
                for pos in positions:
                    # Find matching timeline entry
                    timeline_entry = next((item for item in timeline_data if item['uid4'] == pos['uid']), None)
                    
                    if timeline_entry:
                        f.write(f"{pos['uid']}|{pos['char_pos']}|{pos['char_pos_ub']}|"
                               f"{timeline_entry['mention']}|{timeline_entry['time']}|"
                               f"{timeline_entry['bounds']}|{timeline_entry['known']}|"
                               f"{timeline_entry['context_uid4s']}\n")
                    else:
                        f.write(f"{pos['uid']}|{pos['char_pos']}|{pos['char_pos_ub']}|||\n")
    else:
        # Original output without timeline
        if has_nonempty_file(output_file):
            print(f"Skipping existing charpos output: {output_file}")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            if join_mention:
                f.write("uid4|char_pos|char_pos_ub|mention\n")
            else:
                f.write("uid4|char_pos|char_pos_ub\n")
            
            for pos in positions:
                if join_mention and pos['uid'] in mentions:
                    f.write(f"{pos['uid']}|{pos['char_pos']}|{pos['char_pos_ub']}|{mentions[pos['uid']]}\n")
                else:
                    f.write(f"{pos['uid']}|{pos['char_pos']}|{pos['char_pos_ub']}\n")

def main():
    parser = argparse.ArgumentParser(description='Extract tag positions from text files')
    parser.add_argument('-i', '--input-dir', required=True, help='Input directory')
    parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    parser.add_argument('-p', '--pattern', default='*.txt', help='File pattern to match')
    parser.add_argument('-j', '--join-mention', action='store_true', help='Join with mention file')
    parser.add_argument('-t', '--timeline-dir', help='Timeline directory for joining timeline data')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    timeline_path = Path(args.timeline_dir) if args.timeline_dir else None
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    for input_file in input_path.glob(args.pattern):
        # Determine the corresponding mention file if join_mention is enabled
        mention_file = None
        if args.join_mention:
            base_name = input_file.stem.replace('.txt', '')
            mention_file = input_path / f"{base_name}_uid4.bsv"
        
        output_file = output_path / (input_file.stem + '_positions.bsv')
        
        try:
            process_file(input_file, output_file, args.join_mention, mention_file, timeline_path, output_path)
            print(f"Processed: {input_file.name} using mention: {mention_file}, timeline: {timeline_path}")
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")

if __name__ == '__main__':
    main()
    # Force exit to prevent any potential hanging
    import os
    os._exit(0)
