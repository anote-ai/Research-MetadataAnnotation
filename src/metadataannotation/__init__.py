"""Metadata Annotation research package."""

from metadataannotation.core import (
    DocumentDomain,
    StructureComplexity,
    MetadataAnnotation,
    MetadataExperiment,
    compute_complexity_score,
    StructurePredictor,
)

__all__ = [
    "DocumentDomain",
    "StructureComplexity",
    "MetadataAnnotation",
    "MetadataExperiment",
    "compute_complexity_score",
    "StructurePredictor",
]

__version__ = "0.1.0"
