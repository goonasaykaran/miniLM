import json

def extract_chunks(data, prefix=""):
    """
    Recursively extracts structured text chunks from hierarchical JSON.
    Maintains parent-child relationships to preserve context.
    """
    text_chunks = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix} {key}: " if prefix else f"{key}: "
            text_chunks.extend(extract_chunks(value, new_prefix))
    
    elif isinstance(data, list):
        for item in data:
            text_chunks.extend(extract_chunks(item, prefix))
    
    else:
        chunk = f"{prefix}{data}"
        text_chunks.append(chunk.strip())

    return text_chunks


# Example usage
if __name__ == "__main__":
    # Load hierarchical JSON
    with open("data/sample.json", "r") as file:
        json_data = json.load(file)

    # Extract structured text chunks
    text_chunks = extract_chunks(json_data)

    # Print extracted chunks
    print("\nExtracted Chunks:\n", text_chunks)
