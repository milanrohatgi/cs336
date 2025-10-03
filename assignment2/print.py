import os

# Get all .out files in the current directory
out_files = [f for f in os.listdir('.') if f.endswith('.out')]

# Read and print the contents
for file_name in out_files:
    with open(file_name, 'r') as f:
        contents = f.read()
    print(f"{file_name}: {contents}\n")
