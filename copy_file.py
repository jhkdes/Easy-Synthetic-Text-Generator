import argparse

def copy_file(input_filename, output_filename):
    try:
        with open(input_filename, 'r', encoding='utf-8') as input_file:
            content = input_file.read()
        
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            output_file.write(content)
        
        print(f"File '{input_filename}' successfully copied to '{output_filename}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' does not exist.")
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Copy a file from input to output.")
    parser.add_argument('-i', '--input_filename', required=True, help="Path to the input file.")
    parser.add_argument('-o', '--output_filename', required=True, help="Path to the output file.")
    
    args = parser.parse_args()
    copy_file(args.input_filename, args.output_filename)

if __name__ == "__main__":
    main()
