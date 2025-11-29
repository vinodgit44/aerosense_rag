from rag_pipeline.data_ingestion import load_manual_pdfs, load_telemetry_files
from rag_pipeline.chunking import chunk_text
from rag_pipeline.vector_store import build_collection

print("\n=== BUILDING UAV RAG INDEX ===\n")

# ------------------------------
# 1. Manual ingestion
# ------------------------------
print("[1] Loading manuals...")
manual_docs = load_manual_pdfs()
manual_chunks = chunk_text(manual_docs)

for i, c in enumerate(manual_chunks):
    c["metadata"]["id"] = f"manual_{i}"
    c["metadata"]["source"] = c["metadata"].get("source", "manual")

print(f"✔ Manual chunks: {len(manual_chunks)}")


# ------------------------------
# 2. Telemetry ingestion
# ------------------------------
print("\n[2] Loading telemetry...")
telemetry_records = load_telemetry_files()

# IMPORTANT — chunk telemetry records for RAG indexing
telemetry_chunks = chunk_text(telemetry_records)

for i, c in enumerate(telemetry_chunks):
    c["metadata"]["id"] = f"telemetry_{i}"
    c["metadata"]["source"] = c["metadata"].get("source", "telemetry")

print(f"✔ Telemetry chunks: {len(telemetry_chunks)}")


# ------------------------------
# 3. Build collections
# ------------------------------
print("\n[3] Building Chroma collections...\n")

print("➡ Building MANUAL collection...")
build_collection("manual_chunks", manual_chunks, prefix="manual")

print("\n➡ Building TELEMETRY collection...")
build_collection("telemetry_records", telemetry_chunks, prefix="telemetry")

print("\n🎉 Vector DB built successfully!\n")

