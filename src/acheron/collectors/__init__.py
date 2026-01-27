"""Paper collectors for various academic sources."""

from acheron.collectors.arxiv import ArxivCollector
from acheron.collectors.biorxiv import BiorxivCollector
from acheron.collectors.physionet import PhysioNetCollector
from acheron.collectors.pubmed import PubMedCollector

__all__ = ["PubMedCollector", "BiorxivCollector", "ArxivCollector", "PhysioNetCollector"]
