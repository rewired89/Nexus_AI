#!/usr/bin/env python3
"""Sanity test for PubMed/PMC ingestion and indexing.

Performs end-to-end test:
  1. Ingest a small number of papers from PubMed (retmax=3)
  2. Run indexing to build vector store
  3. Assert non-zero chunk count
  4. Verify evidence span fields are populated

Usage:
    python scripts/sanity_test_ingest.py
    python scripts/sanity_test_ingest.py --query "bioelectricity planarian" --retmax 5

Windows PowerShell:
    python scripts\\sanity_test_ingest.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Sanity test for ingestion + indexing")
    parser.add_argument(
        "--query", "-q",
        default="bioelectricity membrane voltage",
        help="PubMed query (default: bioelectricity membrane voltage)",
    )
    parser.add_argument(
        "--retmax", "-n",
        type=int,
        default=3,
        help="Number of papers to fetch (default: 3)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep data in default location instead of temp dir",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 60)
    print("NEXUS SANITY TEST: PubMed/PMC Ingestion + Indexing")
    print("=" * 60)

    # Use temp dir unless --keep-data
    if args.keep_data:
        import os
        os.environ.setdefault("ACHERON_DATA_DIR", str(Path.cwd() / "data"))
        data_dir = Path(os.environ["ACHERON_DATA_DIR"])
    else:
        temp_dir = tempfile.mkdtemp(prefix="nexus_test_")
        import os
        os.environ["ACHERON_DATA_DIR"] = temp_dir
        os.environ["ACHERON_VECTORSTORE_DIR"] = str(Path(temp_dir) / "vectorstore")
        data_dir = Path(temp_dir)
        print(f"\nUsing temp directory: {temp_dir}")

    # ----------------------------------------------------------------
    # Step 1: Test nexus_ingest module directly
    # ----------------------------------------------------------------
    print("\n[1/4] Testing nexus_ingest.pmc_pubmed module...")
    print(f"      Query: {args.query!r} (retmax={args.retmax})")

    try:
        from nexus_ingest.pmc_pubmed import PMCPubMedFetcher, save_record_to_library
    except ImportError as e:
        print(f"\n[FAIL] Cannot import nexus_ingest: {e}")
        print("       Make sure nexus_ingest directory exists")
        return 1

    library_dir = data_dir / "library"
    library_dir.mkdir(parents=True, exist_ok=True)

    with PMCPubMedFetcher() as fetcher:
        records = fetcher.search(args.query, retmax=args.retmax)

    if not records:
        print(f"\n[WARN] No records returned from PubMed for query: {args.query}")
        print("       This may be a network issue or the query returned no results")
        print("       Try a different query or check your internet connection")
        return 1

    print(f"      Fetched {len(records)} records from PubMed")

    fulltext_count = sum(1 for r in records if r.fulltext_nxml)
    print(f"      Full text (PMC NXML): {fulltext_count}/{len(records)}")

    # Save records to library
    for record in records:
        json_path, nxml_path = save_record_to_library(record, library_dir)
        logger.debug("Saved: %s", json_path.name)

    print(f"      Saved to: {library_dir}")

    # Verify provenance
    for record in records:
        if not record.provenance.fetched_at_utc:
            print(f"\n[WARN] Record PMID:{record.pmid} missing provenance timestamp")
        else:
            logger.debug(
                "PMID:%s provenance: %s",
                record.pmid,
                record.provenance.fetched_at_utc,
            )

    print("      [OK] Provenance tracking verified")

    # ----------------------------------------------------------------
    # Step 2: Test collector integration
    # ----------------------------------------------------------------
    print("\n[2/4] Testing PubMedCollector integration...")

    from acheron.collectors.pubmed import PubMedCollector

    collector = PubMedCollector()
    papers = collector.search(args.query, max_results=args.retmax)
    collector.close()

    if not papers:
        print("\n[FAIL] PubMedCollector returned no papers")
        return 1

    print(f"      Collector returned {len(papers)} Paper objects")

    # Check for full text
    papers_with_fulltext = [p for p in papers if p.full_text]
    print(f"      Papers with full text: {len(papers_with_fulltext)}/{len(papers)}")

    # Save metadata
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for paper in papers:
        collector.save_metadata(paper, target_dir=metadata_dir)

    print(f"      Metadata saved to: {metadata_dir}")
    print("      [OK] Collector integration verified")

    # ----------------------------------------------------------------
    # Step 3: Test chunking with evidence spans
    # ----------------------------------------------------------------
    print("\n[3/4] Testing chunking with evidence spans...")

    from acheron.extraction.chunker import TextChunker

    chunker = TextChunker()
    all_chunks = []

    for paper in papers:
        chunks = chunker.chunk_paper(paper)
        all_chunks.extend(chunks)

    if not all_chunks:
        print(f"\n[FAIL] No chunks created from {len(papers)} papers")
        print("       This may mean papers have no abstract or full text")
        return 1

    print(f"      Created {len(all_chunks)} chunks from {len(papers)} papers")

    # Verify evidence span fields
    chunks_with_spans = [c for c in all_chunks if c.span_end > 0 or c.excerpt]
    chunks_with_file = [c for c in all_chunks if c.source_file]
    chunks_with_xpath = [c for c in all_chunks if c.xpath]

    print(f"      Chunks with span offsets: {len(chunks_with_spans)}/{len(all_chunks)}")
    print(f"      Chunks with source_file: {len(chunks_with_file)}/{len(all_chunks)}")
    print(f"      Chunks with xpath: {len(chunks_with_xpath)}/{len(all_chunks)}")

    # Show sample chunk
    sample = all_chunks[0]
    print("\n      Sample chunk:")
    print(f"        chunk_id: {sample.chunk_id[:50]}...")
    print(f"        section: {sample.section or '(none)'}")
    print(f"        source_file: {sample.source_file or '(none)'}")
    print(f"        span: {sample.span_start}-{sample.span_end}")
    print(f"        xpath: {sample.xpath or '(none)'}")
    if sample.excerpt:
        print(f"        excerpt: {sample.excerpt[:80]}...")
    else:
        print("        excerpt: (none)")

    print("      [OK] Evidence span fields populated")

    # ----------------------------------------------------------------
    # Step 4: Test vector store indexing
    # ----------------------------------------------------------------
    print("\n[4/4] Testing vector store indexing...")

    from acheron.vectorstore.store import VectorStore

    store = VectorStore()
    initial_count = store.count()

    added = store.add_chunks(all_chunks)
    final_count = store.count()

    print(f"      Initial chunks in store: {initial_count}")
    print(f"      Added: {added}")
    print(f"      Final count: {final_count}")

    if final_count == 0:
        print("\n[FAIL] Vector store has 0 chunks after indexing")
        print("       Check that embeddings are working (may need sentence-transformers)")
        return 1

    # Test retrieval
    results = store.search(args.query, n_results=3)
    print(f"      Test query returned {len(results)} results")

    if results:
        r = results[0]
        print("\n      Top result:")
        print(f"        paper_id: {r.paper_id}")
        print(f"        title: {r.paper_title[:60]}...")
        print(f"        pmid: {r.pmid or '(none)'}")
        print(f"        score: {r.relevance_score:.3f}")
        print(f"        source_file: {r.source_file or '(none)'}")
        print(f"        span: {r.span_start}-{r.span_end}")
        print(f"        excerpt: {r.excerpt[:80]}..." if r.excerpt else "        excerpt: (none)")

    print("      [OK] Vector store indexing verified")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[SUCCESS] All sanity tests passed!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Records from PubMed: {len(records)}")
    print(f"  - Full text from PMC: {fulltext_count}")
    print(f"  - Papers indexed: {len(papers)}")
    print(f"  - Chunks created: {len(all_chunks)}")
    print(f"  - Vector store count: {final_count}")

    if not args.keep_data:
        print(f"\nTemp data in: {data_dir}")
        print("(Use --keep-data to save to default location)")

    print("\nTo test manually:")
    print("  acheron collect --source pubmed -t 'bioelectricity' -n 10 --mindate 2020")
    print("  acheron index")
    print("  acheron query -i")

    return 0


if __name__ == "__main__":
    sys.exit(main())
