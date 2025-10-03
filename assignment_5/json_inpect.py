def count_jsonl_entries(filepath):    
    count = 0
    with open(filepath, 'r') as file:
        for line in file:
            count += 1
    return count

filepath = "/data/a5-alignment/MATH/sft.jsonl"
entry_count = count_jsonl_entries(filepath)
print("Number of entries:", entry_count)
