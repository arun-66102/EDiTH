"""
RAG Retriever — Retrieves de-identified historical cases similar to the
current DR diagnosis from the ChromaDB knowledge base.

Uses cosine similarity on sentence-transformer embeddings, with optional
metadata filtering by DR grade.
"""

import logging
from typing import Optional

import chromadb

from app.config import Settings
from app.rag.knowledge_base import build_knowledge_base

logger = logging.getLogger("edith.retriever")


def load_retriever(settings: Settings) -> dict:
    """
    Initialize the RAG retriever by loading/building the ChromaDB collection.

    Returns a dict with the collection for use in retrieval.
    """
    collection = build_knowledge_base(settings)

    return {
        "collection": collection,
        "settings": settings,
    }


def retrieve_similar_cases(
    retriever: dict,
    dr_grade: int,
    top_k: int = 3,
    grade_filter: bool = True,
) -> list[dict]:
    """
    Retrieve the top-K most similar de-identified cases from the knowledge base.

    Args:
        retriever: Dict with 'collection' and 'settings' keys.
        dr_grade: Current patient's DR grade (0–4).
        top_k: Number of cases to retrieve.
        grade_filter: If True, filters cases to the same DR grade ± 1.

    Returns:
        List of dicts with:
            - case_id (str)
            - text (str): Full case description.
            - similarity (float): Cosine similarity score.
            - dr_grade (int): Grade of the retrieved case.
    """
    collection: chromadb.Collection = retriever["collection"]
    settings: Settings = retriever["settings"]

    if collection.count() == 0:
        logger.warning("Knowledge base is empty — no cases to retrieve")
        return []

    # Build query text
    dr_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    query_text = (
        f"Diabetic retinopathy grade {dr_grade} ({dr_labels[dr_grade]}). "
        f"Looking for similar cases with comparable severity and lesion patterns."
    )

    # Build metadata filter for grade ± 1 range
    where_filter: Optional[dict] = None
    if grade_filter:
        adjacent_grades = [
            g for g in range(max(0, dr_grade - 1), min(5, dr_grade + 2))
        ]
        if len(adjacent_grades) == 1:
            where_filter = {"dr_grade": adjacent_grades[0]}
        else:
            where_filter = {
                "$or": [{"dr_grade": g} for g in adjacent_grades]
            }

    # Query ChromaDB
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=min(top_k, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        # Retry without filter
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e2:
            logger.error(f"ChromaDB query failed again: {e2}")
            return []

    # Parse results
    similar_cases = []
    if results and results.get("documents") and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 0.0

            # ChromaDB returns distances (lower = more similar for cosine)
            # Convert to similarity score
            similarity = max(0.0, 1.0 - distance)

            similar_cases.append({
                "case_id": metadata.get("case_id", f"case_{i}"),
                "text": doc,
                "similarity": round(similarity, 4),
                "dr_grade": metadata.get("dr_grade", -1),
            })

    logger.info(f"Retrieved {len(similar_cases)} similar cases for grade {dr_grade}")
    return similar_cases
