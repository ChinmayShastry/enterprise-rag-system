"""
rag/access.py
Document-level access control.

Roles used to differ only in how many chunks they got back, which is not an
access boundary — every role searched the same unfiltered corpus, so a viewer
and an admin could surface the same confidential text. AccessPolicy makes
clearance decide *which* documents a role can retrieve at all.

The policy is applied in three places (see rag/retrieval.py):

  1. Inside the vector store query, so denied chunks are never returned.
  2. While selecting BM25 candidates, which search an in-memory list that
     Chroma's filter cannot reach.
  3. Once more on the final document set, so a bug in either search path
     cannot leak a chunk into the prompt.

Steps 1 and 2 are the enforcement; step 3 is the backstop.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag.settings import RolePermissions, Settings

# Metadata key holding a chunk's classification label, written by ingest.py.
CLASSIFICATION_KEY = "classification"


@dataclass(frozen=True)
class AccessPolicy:
    """
    What one role is allowed to retrieve.

    Build it with `for_role()` rather than constructing it directly, so the
    clearance always comes from config rather than a call site.
    """

    role: str
    clearance: frozenset[str]

    @classmethod
    def for_role(cls, role: str, settings: Settings) -> "AccessPolicy":
        return cls(role=role, clearance=settings.permissions_for(role).clearance)

    @classmethod
    def from_permissions(cls, role: str, permissions: RolePermissions) -> "AccessPolicy":
        return cls(role=role, clearance=permissions.clearance)

    @property
    def denies_everything(self) -> bool:
        return not self.clearance

    # ── enforcement ──────────────────────────────────────────────

    def permits(self, doc: Document) -> bool:
        """
        Whether this role may see a chunk.

        Deny-by-default: a chunk with no classification, or one carrying a
        label this deployment does not recognise, belongs to nobody. Such
        chunks predate labelling and should be fixed by re-ingesting, not by
        widening a clearance list.
        """
        label = doc.metadata.get(CLASSIFICATION_KEY)
        return isinstance(label, str) and label in self.clearance

    def filter(self, docs: list[Document]) -> list[Document]:
        """Drop every chunk this role may not see, preserving order."""
        return [doc for doc in docs if self.permits(doc)]

    def where_clause(self) -> dict | None:
        """
        Chroma metadata filter for this role's clearance.

        Returns None when the role is cleared for nothing — callers must treat
        that as "retrieve nothing" rather than passing None to Chroma, which
        would mean "no filter" and return everything.
        """
        if not self.clearance:
            return None
        return {CLASSIFICATION_KEY: {"$in": sorted(self.clearance)}}


def unlabelled(docs: list[Document]) -> list[Document]:
    """
    Chunks carrying no classification. These are invisible to every role;
    the app surfaces a count so the operator knows to re-ingest.
    """
    return [
        doc
        for doc in docs
        if not isinstance(doc.metadata.get(CLASSIFICATION_KEY), str)
    ]
