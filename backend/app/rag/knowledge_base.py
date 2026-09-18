"""
RAG Knowledge Base — Ingests de-identified clinical cases and guidelines
into a ChromaDB vector store for similarity-based retrieval.

Cases are embedded using sentence-transformers and stored with metadata
(DR grade, lesion types) to enable filtered retrieval.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import Settings

logger = logging.getLogger("edith.knowledge_base")


def load_cases(data_dir: Path) -> list[dict]:
    """
    Load de-identified case data from JSON file.

    Expected format: list of dicts with keys:
        - case_id (str)
        - dr_grade (int): 0–4
        - lesion_types (list[str])
        - clinical_notes (str)
        - treatment (str)
        - validated_report (str)
    """
    cases_path = data_dir / "cases" / "cases.json"
    if not cases_path.exists():
        logger.warning(f"Cases file not found at {cases_path}")
        return []

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    logger.info(f"Loaded {len(cases)} cases from {cases_path}")
    return cases


def load_guidelines(data_dir: Path) -> str:
    """Load clinical DR guidelines as a single text block."""
    guidelines_path = data_dir / "clinical_guidelines" / "dr_guidelines.md"
    if not guidelines_path.exists():
        logger.warning(f"Guidelines file not found at {guidelines_path}")
        return ""

    with open(guidelines_path, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Loaded guidelines ({len(text)} chars)")
    return text


def build_knowledge_base(settings: Settings) -> chromadb.Collection:
    """
    Build or load the ChromaDB collection with embedded case data.

    If the collection already exists and has data, returns it directly.
    Otherwise, ingests cases and guidelines from the data directory.
    """
    # Initialize ChromaDB with persistent storage
    settings.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))

    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=settings.RAG_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Check if already populated
    if collection.count() > 0:
        logger.info(
            f"Knowledge base already populated ({collection.count()} documents)"
        )
        return collection

    # Load embedding model
    embedder = SentenceTransformer(settings.RAG_EMBEDDING_MODEL)

    # --- Ingest cases ---
    cases = load_cases(settings.DATA_DIR)

    if cases:
        documents = []
        metadatas = []
        ids = []

        for case in cases:
            # Create a rich text representation for embedding
            doc_text = (
                f"DR Grade: {case['dr_grade']} "
                f"({['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'][case['dr_grade']]}). "
                f"Lesions: {', '.join(case.get('lesion_types', ['none']))}. "
                f"Clinical Notes: {case.get('clinical_notes', 'N/A')}. "
                f"Treatment: {case.get('treatment', 'N/A')}. "
                f"Report: {case.get('validated_report', 'N/A')}"
            )

            documents.append(doc_text)
            metadatas.append({
                "case_id": case["case_id"],
                "dr_grade": case["dr_grade"],
                "lesion_types": ", ".join(case.get("lesion_types", [])),
            })
            ids.append(case["case_id"])

        # Batch add to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Ingested {len(cases)} cases into knowledge base")

    # --- Ingest guidelines ---
    guidelines = load_guidelines(settings.DATA_DIR)
    if guidelines:
        # Split guidelines into chunks (~500 chars each)
        chunks = []
        current_chunk = ""
        for line in guidelines.split("\n"):
            if len(current_chunk) + len(line) > 500 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += "\n" + line
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if chunks:
            collection.add(
                documents=chunks,
                metadatas=[
                    {"case_id": f"guideline_{i}", "dr_grade": -1, "lesion_types": ""}
                    for i in range(len(chunks))
                ],
                ids=[f"guideline_{i}" for i in range(len(chunks))],
            )
            logger.info(f"Ingested {len(chunks)} guideline chunks into knowledge base")

    return collection
