from __future__ import annotations

import argparse

from src.retriever import ChromaRetriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the Chroma collection.")
    args = parser.parse_args()
    retriever = ChromaRetriever(rebuild=args.rebuild)
    print(f"Chroma collection ready with {retriever.collection.count()} chunks.")
