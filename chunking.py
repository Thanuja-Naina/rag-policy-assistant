"""Step 1: split each policy file into small pieces ("chunks").

Each '## Heading' section of a markdown file becomes one chunk.
Small chunks make search more precise than searching whole documents.
"""
from pathlib import Path


def split_into_chunks(text, file_name):
    """Turn one markdown text into a list of chunks (dictionaries)."""
    chunks = []
    title = text.splitlines()[0].replace("# ", "").strip()  # e.g. "Leave Policy"
    sections = text.split("\n## ")[1:]                        # everything after each '## '

    for section in sections:
        heading, _, body = section.partition("\n")            # first line = heading
        body = " ".join(body.split())                         # remove extra spaces/line breaks
        chunks.append({
            "text": f"{title} - {heading.strip()}: {body}",
            "source": file_name,                              # e.g. leave_policy.md
            "section": heading.strip(),                       # e.g. Annual Leave
        })
    return chunks


def load_all_chunks(folder="data/policies"):
    """Read every .md file in the folder and return all chunks."""
    all_chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(split_into_chunks(text, path.name))
    return all_chunks
