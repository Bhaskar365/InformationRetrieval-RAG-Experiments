
import json
from pathlib import Path

print("DEBUG: chunk loader running")

def load_chunks(path: str | Path) -> list[dict]:

    path = Path(path)
    print("DEBUG chunk path:", path)

    chunks = []

    with path.open("r",encoding="utf-8") as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            chunks.append(json.loads(line))

    return chunks


def build_chunk_lookup(
    chunks: list[dict]
) -> dict:

    return {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }
