import argparse, os, json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    for pdf in sorted(input_dir.rglob("*.pdf")):
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    if not docs:
        print("No PDFs found in", input_dir); return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # OpenAI-compatible embeddings (works with Azure/OpenAI); swap out to SentenceTransformer if you prefer
    embeddings = OpenAIEmbeddings(model=args.embed_model)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_dir))
    print("Saved FAISS index to", index_dir)

if __name__ == "__main__":
    main()
