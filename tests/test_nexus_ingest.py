"""Offline tests for nexus_ingest.pmc_pubmed module.

Tests XML parsing, NXML section extraction, library saving, and provenance
without any network calls.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from nexus_ingest.pmc_pubmed import (
    PMCPubMedFetcher,
    PubMedRecord,
    SourceProvenance,
    save_record_to_library,
)

# ======================================================================
# Sample XML fixtures
# ======================================================================
SAMPLE_PUBMED_XML = """\
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Bioelectric signaling in planarian regeneration</ArticleTitle>
        <Abstract>
          <AbstractText>Bioelectric signals regulate head-tail polarity in planaria.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <ForeName>Michael</ForeName>
            <LastName>Levin</LastName>
          </Author>
          <Author>
            <ForeName>Dany S</ForeName>
            <LastName>Adams</LastName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>Nature Communications</Title>
          <JournalIssue>
            <PubDate>
              <Year>2023</Year>
              <Month>06</Month>
              <Day>15</Day>
            </PubDate>
          </JournalIssue>
        </Journal>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName>Planarians</DescriptorName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName>Regeneration</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
      <KeywordList>
        <Keyword>bioelectricity</Keyword>
        <Keyword>Vmem</Keyword>
      </KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1038/s41467-023-99999-0</ArticleId>
        <ArticleId IdType="pmc">PMC9876543</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

SAMPLE_NXML = """\
<article>
  <front>
    <article-meta>
      <abstract>
        <p>Bioelectric signals play a key role in planarian regeneration.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>Planarians are flatworms with remarkable regenerative ability.</p>
      <p>They can regrow a head from a tail fragment.</p>
    </sec>
    <sec>
      <title>Results</title>
      <p>Vmem gradients were measured across the anterior-posterior axis.</p>
      <p>Depolarization of anterior tissue blocked head regeneration.</p>
    </sec>
    <sec>
      <title>Discussion</title>
      <p>These findings support the bioelectric model of pattern regulation.</p>
    </sec>
  </body>
</article>
"""


# ======================================================================
# SourceProvenance tests
# ======================================================================
def test_source_provenance_defaults():
    prov = SourceProvenance()
    assert prov.provider == "NCBI"
    assert prov.database == "pubmed"
    assert prov.fetched_at_utc == ""
    assert prov.request_query == ""
    assert prov.url_used == ""


def test_source_provenance_custom():
    prov = SourceProvenance(
        provider="NCBI",
        database="pmc",
        fetched_at_utc="2025-06-01T12:00:00+00:00",
        request_query="bioelectricity planarian",
        url_used="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc",
    )
    assert prov.database == "pmc"
    assert "2025" in prov.fetched_at_utc
    assert "bioelectricity" in prov.request_query


# ======================================================================
# PubMedRecord tests
# ======================================================================
def test_pubmed_record_defaults():
    rec = PubMedRecord(pmid="12345")
    assert rec.pmid == "12345"
    assert rec.pmcid is None
    assert rec.doi is None
    assert rec.authors == []
    assert rec.keywords == []
    assert rec.mesh_terms == []
    assert rec.fulltext_nxml is None
    assert rec.fulltext_sections == []


def test_pubmed_record_to_dict():
    rec = PubMedRecord(
        pmid="12345",
        pmcid="PMC67890",
        doi="10.1234/test",
        title="Test Paper",
        journal="Nature",
        year=2023,
        pub_date="2023-06-15",
        authors=["Levin M", "Adams DS"],
        abstract="Test abstract",
        keywords=["bioelectricity"],
        mesh_terms=["Planarians"],
        fulltext_nxml="<article>...</article>",
        fulltext_sections=[{"heading": "Intro", "text": "Body text"}],
        provenance=SourceProvenance(
            provider="NCBI",
            database="pubmed",
            fetched_at_utc="2025-01-01T00:00:00Z",
            request_query="test query",
            url_used="https://example.com",
        ),
    )
    d = rec.to_dict()

    assert d["pmid"] == "12345"
    assert d["pmcid"] == "PMC67890"
    assert d["doi"] == "10.1234/test"
    assert d["has_fulltext"] is True
    assert len(d["fulltext_sections"]) == 1
    assert d["provenance"]["provider"] == "NCBI"
    assert d["provenance"]["fetched_at_utc"] == "2025-01-01T00:00:00Z"
    assert d["provenance"]["request_query"] == "test query"


def test_pubmed_record_to_dict_no_fulltext():
    rec = PubMedRecord(pmid="99999", title="No Fulltext Paper")
    d = rec.to_dict()
    assert d["has_fulltext"] is False
    assert d["fulltext_sections"] == []


# ======================================================================
# XML parsing tests (offline, no network)
# ======================================================================
def test_parse_pubmed_article():
    """Test parsing a PubMed XML article element."""
    fetcher = PMCPubMedFetcher.__new__(PMCPubMedFetcher)
    root = ET.fromstring(SAMPLE_PUBMED_XML)
    article_el = root.find(".//PubmedArticle")

    record = fetcher._parse_pubmed_article(
        article_el,
        query="bioelectricity planarian",
        url="https://example.com/efetch",
    )

    assert record is not None
    assert record.pmid == "12345678"
    assert record.pmcid == "PMC9876543"
    assert record.doi == "10.1038/s41467-023-99999-0"
    assert "Bioelectric signaling" in record.title
    assert "planaria" in record.abstract
    assert len(record.authors) == 2
    assert "Michael Levin" in record.authors
    assert "Dany S Adams" in record.authors
    assert record.year == 2023
    assert record.pub_date == "2023-06-15"
    assert record.journal == "Nature Communications"
    assert "Planarians" in record.mesh_terms
    assert "bioelectricity" in record.keywords
    # Provenance should be set
    assert record.provenance.provider == "NCBI"
    assert record.provenance.database == "pubmed"
    assert record.provenance.fetched_at_utc  # non-empty
    assert record.provenance.request_query == "bioelectricity planarian"


def test_parse_pubmed_article_missing_medline():
    """Test parsing an article element with no MedlineCitation."""
    fetcher = PMCPubMedFetcher.__new__(PMCPubMedFetcher)
    xml = "<PubmedArticle><PubmedData></PubmedData></PubmedArticle>"
    article_el = ET.fromstring(xml)
    result = fetcher._parse_pubmed_article(article_el, "test", "url")
    assert result is None


def test_parse_pubmed_article_no_pmid():
    """Test parsing an article with no PMID."""
    fetcher = PMCPubMedFetcher.__new__(PMCPubMedFetcher)
    xml = "<PubmedArticle><MedlineCitation><Article><ArticleTitle>Test</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
    article_el = ET.fromstring(xml)
    result = fetcher._parse_pubmed_article(article_el, "test", "url")
    assert result is None


# ======================================================================
# NXML section parsing tests
# ======================================================================
def test_parse_nxml_sections():
    """Test parsing NXML body sections."""
    fetcher = PMCPubMedFetcher.__new__(PMCPubMedFetcher)
    sections = fetcher._parse_nxml_sections(SAMPLE_NXML)

    # Should have: Abstract + Introduction + Results + Discussion
    assert len(sections) >= 3
    headings = [s["heading"] for s in sections]
    assert "Introduction" in headings
    assert "Results" in headings
    assert "Discussion" in headings

    # Check section content
    intro = next(s for s in sections if s["heading"] == "Introduction")
    assert "flatworms" in intro["text"]
    assert "regrow a head" in intro["text"]

    results = next(s for s in sections if s["heading"] == "Results")
    assert "Vmem gradients" in results["text"]

    # Check xpath tracking
    for sec in sections:
        assert "xpath" in sec
        if sec["heading"] == "Results":
            assert "Results" in sec["xpath"]


def test_parse_nxml_sections_empty():
    """Test parsing empty or invalid NXML."""
    fetcher = PMCPubMedFetcher.__new__(PMCPubMedFetcher)

    # Invalid XML
    sections = fetcher._parse_nxml_sections("<not-valid-xml")
    assert sections == []


def test_parse_nxml_sections_no_body():
    """Test parsing NXML with no body element."""
    fetcher = PMCPubMedFetcher.__new__(PMCPubMedFetcher)
    nxml = "<article><front><article-meta></article-meta></front></article>"
    sections = fetcher._parse_nxml_sections(nxml)
    # Should return empty or just abstract (no body sections)
    assert len(sections) == 0


def test_parse_nxml_abstract_extraction():
    """Test that abstract is extracted from NXML front matter."""
    fetcher = PMCPubMedFetcher.__new__(PMCPubMedFetcher)
    sections = fetcher._parse_nxml_sections(SAMPLE_NXML)

    abstract_sections = [s for s in sections if s["heading"] == "Abstract"]
    assert len(abstract_sections) == 1
    assert "planarian regeneration" in abstract_sections[0]["text"]
    assert abstract_sections[0]["xpath"] == "front/article-meta/abstract"


# ======================================================================
# Library saving tests
# ======================================================================
def test_save_record_to_library(tmp_path):
    """Test saving a record with metadata JSON."""
    record = PubMedRecord(
        pmid="11111",
        title="Test Paper for Library",
        authors=["Author A"],
        abstract="Test abstract content",
        provenance=SourceProvenance(
            provider="NCBI",
            database="pubmed",
            fetched_at_utc="2025-01-01T00:00:00Z",
            request_query="test query",
        ),
    )

    json_path, nxml_path = save_record_to_library(record, tmp_path)

    assert json_path.exists()
    assert json_path.name == "pubmed_11111.json"
    assert nxml_path is None  # No NXML for this record

    # Verify JSON content
    data = json.loads(json_path.read_text())
    assert data["pmid"] == "11111"
    assert data["title"] == "Test Paper for Library"
    assert data["provenance"]["provider"] == "NCBI"
    assert data["provenance"]["fetched_at_utc"] == "2025-01-01T00:00:00Z"


def test_save_record_to_library_with_nxml(tmp_path):
    """Test saving a record with both JSON and NXML."""
    record = PubMedRecord(
        pmid="22222",
        pmcid="PMC99999",
        title="Paper with Full Text",
        fulltext_nxml=SAMPLE_NXML,
        provenance=SourceProvenance(),
    )

    json_path, nxml_path = save_record_to_library(record, tmp_path)

    assert json_path.exists()
    assert nxml_path is not None
    assert nxml_path.exists()
    assert nxml_path.name == "pmc_PMC99999.nxml"

    # NXML content should be the sample
    nxml_content = nxml_path.read_text()
    assert "<article>" in nxml_content
    assert "Introduction" in nxml_content


def test_save_record_to_library_pmcid_normalization(tmp_path):
    """Test that PMCID is normalized (PMC prefix added if missing)."""
    record = PubMedRecord(
        pmid="33333",
        pmcid="99999",  # no PMC prefix
        fulltext_nxml="<article>test</article>",
        provenance=SourceProvenance(),
    )

    _, nxml_path = save_record_to_library(record, tmp_path)
    assert nxml_path is not None
    assert nxml_path.name == "pmc_PMC99999.nxml"


def test_save_record_creates_directory(tmp_path):
    """Test that save creates the library directory if it doesn't exist."""
    new_dir = tmp_path / "nested" / "library"
    record = PubMedRecord(pmid="44444", provenance=SourceProvenance())

    json_path, _ = save_record_to_library(record, new_dir)
    assert json_path.exists()
    assert new_dir.exists()


# ======================================================================
# Static helper tests
# ======================================================================
def test_get_text_helper():
    """Test the _get_text static method."""
    fetcher = PMCPubMedFetcher.__new__(PMCPubMedFetcher)

    # Normal element
    el = ET.fromstring("<tag>Hello World</tag>")
    assert fetcher._get_text(el) == "Hello World"

    # Nested elements (itertext)
    el = ET.fromstring("<tag>Hello <b>bold</b> text</tag>")
    assert fetcher._get_text(el) == "Hello bold text"

    # None element
    assert fetcher._get_text(None) == ""

    # Empty element
    el = ET.fromstring("<tag></tag>")
    assert fetcher._get_text(el) == ""
