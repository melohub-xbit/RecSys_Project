import json
import argparse
import os

def copy_json_entries(input_path, output_path, num_entries):
    """
    Copies the first `num_entries` from a large JSON/JSONL file.
    Handles both JSONL (one JSON object per line) and JSON array formats.
    """
    if not os.path.exists(input_path):
        print(f"Error: Could not find '{input_path}'")
        return

    print(f"Reading first {num_entries} entries from {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        # Check first non-whitespace character to determine format
        first_char = ''
        while True:
            char = infile.read(1)
            if not char:
                break
            if char.strip():
                first_char = char
                break
        infile.seek(0)
        
        if first_char == '[':
            print("Detected standard JSON array format. Using basic array streaming...")
            outfile.write("[\n")
            
            # Skip the opening bracket
            while True:
                char = infile.read(1)
                if char == '[':
                    break
                    
            # A very simple state machine to extract JSON objects
            # Assumes objects are properly formatted and separated by commas
            entries_found = 0
            brace_count = 0
            in_string = False
            escape_next = False
            current_obj = []
            
            while entries_found < num_entries:
                char = infile.read(1)
                if not char:
                    break
                    
                current_obj.append(char)
                
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"':
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # We finished an object
                            obj_str = ''.join(current_obj)
                            try:
                                # Validate it's proper JSON
                                parsed = json.loads(obj_str)
                                
                                if entries_found > 0:
                                    outfile.write(",\n")
                                json.dump(parsed, outfile, indent=2)
                                entries_found += 1
                                current_obj = [] # Reset for next object
                                
                                # Skip whitespace and comma
                                while True:
                                    next_char = infile.read(1)
                                    if next_char not in (' ', '\n', '\r', '\t', ','):
                                        infile.seek(infile.tell() - 1)
                                        break
                            except json.JSONDecodeError:
                                pass # Incomplete object, keep reading
                                
            outfile.write("\n]\n")
            print(f"Successfully wrote {entries_found} entries to {output_path}")

        else:
            print("Detected JSONL format (one object per line)...")
            count = 0
            for line in infile:
                if not line.strip():
                    continue
                try:
                    # Validate JSON to ensure it's a complete line
                    parsed = json.loads(line)
                    json.dump(parsed, outfile)
                    outfile.write("\n")
                    count += 1
                    if count >= num_entries:
                        break
                except json.JSONDecodeError:
                    print(f"Skipped improperly formatted line {count+1}")
                    
            print(f"Successfully wrote {count} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy first N entries from a large JSON/JSONL file.")
    parser.add_argument("input_file", help="Path to the large input file")
    parser.add_argument("--output", "-o", default="sample_output.json", help="Path for the output file (default: sample_output.json)")
    parser.add_argument("--num", "-n", type=int, default=5, help="Number of entries to extract (default: 5)")
    
    args = parser.parse_args()
    copy_json_entries(args.input_file, args.output, args.num)
