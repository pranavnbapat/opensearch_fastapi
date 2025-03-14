# utils.py

PAGE_SIZE = 10

# Convert all filters to lowercase to match OpenSearch indexing
def lowercase_list(values):
    return [v.lower() for v in values] if values else []