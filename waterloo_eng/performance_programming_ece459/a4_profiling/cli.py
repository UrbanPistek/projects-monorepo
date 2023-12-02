import json
import argparse

def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser.parse_args:
    parser.add_argument(
        "-j",
        "--json", 
        action='store_true',
        help='Read and parse from benchmark.json')
    parser.add_argument(
        "-v",
        "--val", 
        type=str,
        help='Parse input string')
    parser.add_argument(
        "-p",
        "--parse", 
        action='store_true',
        help='Enable Parsing of input string')

    return parser.parse_args()

def main():
    
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    if args.json:

        # Open the JSON file
        with open('benchmark.json') as f:
            data = json.load(f)

        # Get the value of the "name" key
        val = round(data['results'][0]['mean'], 5)*1000 # convert to ms

        # Print the name
        print(f"> time= {val:.3f} ms")

    elif args.parse:
        
        input_str = args.val
        global_checksums = input_str.split("Global checksums:")[1]
        split_s = global_checksums.split()
        for i in range(0, len(split_s), 3):
            print(str(split_s[i] + " " + split_s[i+1] + " " + split_s[i+2]))

if __name__ == "__main__":
    main()

