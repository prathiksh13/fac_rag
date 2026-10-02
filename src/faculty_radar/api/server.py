"""Demo FastAPI server: HTTP wrapper around the existing RagPipeline.

This module contains no retrieval, ranking, or generation logic of its own.
Every number in every response is computed live by the real pipeline stages
(normalization -> resolution -> linking -> corpus -> hybrid retrieval ->
reranking -> expertise -> evidence -> extractive generation -> verification).

Demo corpus: the hermetic OpenAlex-shaped fixtures (4 researchers, 5 works)
run through those real stages in-memory at startup, with an explicit,
hard-coded-for-demo consent registry. Nothing is served from files, nothing
is hardcoded per query.

MODES (FACULTY_API_MODE):
  ondemand (default) - every request discovers live candidates from OpenAlex
      (title/abstract search, institution-scoped), adapts them through the
      real resolution/linking/corpus stages, and runs the unchanged
      RagPipeline. No bulk download; results cached 24h by normalized query.
  corpus   - serve a prebuilt data lake (FACULTY_API_DATA_DIR).
  fixture  - the 4-researcher unit-test fixtures (tests only, never production).

PRODUCTION CORPUS: FACULTY_API_MODE=corpus with FACULTY_API_DATA_DIR=<lake root>,
where <lake root>/data holds a real pipeline output:

    data/interim/      normalized authors/works (InterimStore)
    data/processed/    consent.json + faculty.jsonl (ConsentRegistry/FacultyStore)
    data/corpus/       documents.jsonl (CorpusStore)

Build it by running the real stages against live OpenAlex (scope via
FR_SCOPE__INSTITUTION_IDS / FR_SCOPE__INSTITUTION_NAMES), curating
data/processed/consent.json, then linking + corpus building. Datalake mode
fails fast at startup when those files are absent - it never silently falls
back to fixtures.

Run from the repo root:
    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe -m uvicorn faculty_radar.api.server:app --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Reuses the offline OpenAlex-shaped fixtures so the demo needs no network.
# The payloads enter through the real normalization stage, exactly as a live
# ingestion snapshot would.
from tests.fake_openalex import AUTHORS, WORKS

from faculty_radar.config import get_logger
from faculty_radar.corpus.builder import build_corpus
from faculty_radar.corpus.store import CorpusStore
from faculty_radar.linking.builder import link_faculty
from faculty_radar.linking.consent import ConsentEntry, ConsentRegistry
from faculty_radar.normalization.authors import normalize_authors
from faculty_radar.normalization.pipeline import InterimStore
from faculty_radar.normalization.works import normalize_works
from faculty_radar.rag.pipeline import RagPipeline
from faculty_radar.resolution.resolver import resolve_researchers

logger = get_logger(__name__)

# Explicit demo-only consent. Wei Zhang declares one stated topic and one
# project description; everyone else is consented with no declared topics.
# The Southport homonym (A100000002) is deliberately left OUTSIDE the demo
# institution scope, so searches must never surface marine biology.
DEMO_CONSENT_IDS = ("A100000001", "A100000003", "A100000004")
DEMO_STATED_TOPICS = {"A100000001": ["Molecular Property Prediction"]}
DEMO_STATED_PROJECTS = {
    "A100000001": ["Explain which molecular structures cause a predicted property."]
}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


def api_mode() -> str:
    """Serving mode: 'ondemand' (default), 'corpus', or 'fixture'."""
    return os.environ.get("FACULTY_API_MODE", "ondemand").strip().lower()


def build_datalake_pipeline() -> RagPipeline:
    """Serve a real pipeline data lake: no fixtures, no in-memory shortcuts.

    Reads data/{interim,processed,corpus} written by actual pipeline runs.
    Only consented profiles exist in processed/faculty.jsonl (the linking
    stage guarantees that), so consent is enforced by the data itself.
    """
    from pathlib import Path

    from faculty_radar.config.settings import Settings
    from faculty_radar.linking.store import FacultyStore
    from faculty_radar.paths import build_paths

    data_dir = os.environ.get("FACULTY_API_DATA_DIR", "").strip()
    if not data_dir:
        raise RuntimeError(
            "datalake mode needs FACULTY_API_DATA_DIR=<lake root> "
            "with <lake root>/data/{interim,processed,corpus} populated"
        )
    root = Path(data_dir)
    settings = Settings(
        paths={"root": root, "data": root / "data", "logs": root / "logs"},
        logging={"file_enabled": False, "console_enabled": False},
    )
    paths = build_paths(settings)
    missing = [
        name
        for name, path in (
            ("interim works", paths.interim / "works.jsonl"),
            ("linked faculty", paths.processed / "faculty.jsonl"),
            ("corpus chunks", paths.corpus / "documents.jsonl"),
        )
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "datalake mode found no usable corpus at "
            f"{root}: missing {', '.join(missing)}. Build it with the real "
            "stages (ingest -> normalize -> resolve -> link with consent.json "
            "-> corpus) before starting the server."
        )
    works = InterimStore(paths).load_works()
    profiles = FacultyStore(paths).load_profiles()
    documents = CorpusStore(paths).load_documents()
    if not profiles or not documents:
        raise RuntimeError(
            f"datalake at {root} loaded {len(profiles)} profiles and "
            f"{len(documents)} documents; refusing to serve an empty corpus"
        )
    pipeline = RagPipeline(profiles, works, documents, settings, paths)
    logger.info(
        "datalake pipeline ready",
        extra={
            "root": str(root),
            "profiles": len(profiles),
            "documents": len(documents),
        },
    )
    return pipeline


def build_demo_pipeline() -> RagPipeline:
    """Run the real pipeline stages over the demo fixtures, in memory."""
    import tempfile
    from pathlib import Path

    from faculty_radar.config.settings import Settings
    from faculty_radar.paths import build_paths

    demo_root = Path(tempfile.gettempdir()) / "faculty-radar-demo"
    settings = Settings(
        paths={
            "root": demo_root,
            "data": demo_root / "data",
            "logs": demo_root / "logs",
        },
        logging={"file_enabled": False, "console_enabled": False},
        openalex={"requests_per_second": 0.0, "backoff_seconds": 0.0},
        scope={
            "institution_ids": ["I1000"],
            "institution_names": ["Northgate University"],
            "topic_seeds": ["graph neural networks"],
        },
    )
    authors = normalize_authors(AUTHORS)
    works = normalize_works(WORKS)
    resolved = resolve_researchers(authors, works, settings, store=False)
    consent = ConsentRegistry(
        {
            researcher_id: ConsentEntry(
                researcher_id=researcher_id,
                consented=True,
                stated_topics=DEMO_STATED_TOPICS.get(researcher_id, []),
                stated_project_descriptions=DEMO_STATED_PROJECTS.get(researcher_id, []),
            )
            for researcher_id in DEMO_CONSENT_IDS
        },
        enforce=True,
    )
    profiles = link_faculty(
        resolved.researchers, authors, works, settings, consent=consent, store=False
    )
    documents = build_corpus(profiles, works, settings, store=False)
    pipeline = RagPipeline(profiles, works, documents, settings, build_paths(settings))
    logger.info(
        "demo pipeline ready",
        extra={"profiles": len(profiles), "documents": len(documents)},
    )
    return pipeline


def _match_payload(match, profiles_by_id: dict) -> dict:
    profile = profiles_by_id.get(match.faculty_id)
    stated = [t.name for t in profile.stated_topics] if profile else []
    return {
        "faculty_id": match.faculty_id,
        "faculty_name": match.faculty_name,
        "score": match.score,
        "expertise_type": match.expertise_type.value,
        "stated_score": match.stated_score,
        "evidence_score": match.evidence_score,
        "inferred_score": match.inferred_score,
        "institutions": [{"id": i.id, "name": i.name} for i in match.institutions],
        "stated_topics": stated,
        "matched_topics": match.matched_topics,
        "supporting_publications": [
            {
                "work_id": p.work_id,
                "title": p.title,
                "year": p.year,
                "doi": p.doi,
                "citation_url": p.citation_url,
            }
            for p in match.supporting_publications
        ],
        "evidence": [
            {
                "quote": e.quote,
                "title": e.title,
                "year": e.year,
                "doi": e.doi,
                "source_url": e.source_url,
                "expertise_type": e.expertise_type.value,
                "relevance": e.relevance,
                "work_id": e.work_id,
                "doc_id": e.doc_id,
            }
            for e in match.evidence
        ],
        "signals": [
            {
                "name": s.name,
                "value": s.value,
                "weight": s.weight,
                "contribution": s.contribution,
                "explanation": s.explanation,
            }
            for s in match.signals
        ],
        "notes": match.notes,
    }


def _answer_payload(answer) -> dict:
    return {
        "query": answer.query,
        "status": answer.status.value,
        "verified": answer.verified,
        "answer_text": answer.answer_text,
        "warnings": answer.warnings,
        "llm_model": answer.llm_model,
        "claims": [
            {
                "text": claim.text,
                "expertise_type": claim.expertise_type.value,
                "status": claim.status.value,
                "verification_note": claim.verification_note,
                "citations": [
                    {
                        "evidence_id": c.evidence_id,
                        "title": c.title,
                        "year": c.year,
                        "doi": c.doi,
                        "source_url": c.source_url,
                        "quote": c.quote,
                        "work_id": c.work_id,
                        "faculty_id": c.faculty_id,
                    }
                    for c in claim.citations
                ],
            }
            for claim in answer.claims
        ],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = api_mode()
    app.state.mode = mode
    if mode == "ondemand":
        # No startup build: each request discovers its own candidates live.
        # Fixture/corpus modes prebuild once here.
        from faculty_radar.discovery.service import DiscoveryService

        app.state.discovery = DiscoveryService(cache_dir=_discovery_cache_dir())
        app.state.pipeline = None
        app.state.corpus_source = "live-openalex"
        app.state.profiles_by_id = {}
    elif mode == "corpus":
        pipeline = build_datalake_pipeline()
        app.state.discovery = None
        app.state.pipeline = pipeline
        app.state.corpus_source = "datalake"
        app.state.profiles_by_id = {p.faculty_id: p for p in pipeline.profiles}
    elif mode == "fixture":
        pipeline = build_demo_pipeline()
        app.state.discovery = None
        app.state.pipeline = pipeline
        app.state.corpus_source = "fixture"
        app.state.profiles_by_id = {p.faculty_id: p for p in pipeline.profiles}
    else:
        raise RuntimeError(
            f"unknown FACULTY_API_MODE={mode!r}; expected 'ondemand', 'corpus', or 'fixture'"
        )
    yield


def _discovery_cache_dir():
    from pathlib import Path

    return Path.cwd() / "data" / "cache" / "discovery"


app = FastAPI(title="Faculty Research Discovery (demo)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    pipeline: RagPipeline | None = getattr(app.state, "pipeline", None)
    if pipeline is None and getattr(app.state, "mode", None) != "ondemand":
        raise HTTPException(status_code=503, detail="pipeline not ready")
    base = {
        "status": "ok",
        "mode": getattr(app.state, "mode", "unknown"),
        "corpus_source": getattr(app.state, "corpus_source", "unknown"),
    }
    if pipeline is not None:
        base.update(
            {
                "profiles": len(pipeline.profiles),
                "documents": len(pipeline.documents),
                "llm_model": pipeline.settings.llm.model if pipeline.settings else None,
            }
        )
    else:
        discovery = getattr(app.state, "discovery", None)
        base.update(
            {
                "scope_institutions": discovery.scope_ids if discovery else [],
                "llm_model": "extractive-v1",
            }
        )
    return base


@app.post("/api/faculty/search")
def faculty_search(request: SearchRequest) -> dict:
    mode = getattr(app.state, "mode", "ondemand")
    if mode == "ondemand":
        return _ondemand_search(request)
    pipeline: RagPipeline | None = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    answer = pipeline.answer(request.query.strip(), top_k=request.top_k)
    payload = _answer_payload(answer)
    payload["corpus_source"] = getattr(app.state, "corpus_source", "unknown")
    payload["corpus_profiles"] = len(pipeline.profiles)
    payload["corpus_documents"] = len(pipeline.documents)
    # Matches carry the per-faculty breakdown the cards render; empty when
    # the answer is insufficient_evidence, by construction upstream.
    payload["matches"] = [
        _match_payload(match, app.state.profiles_by_id)
        for match in answer.expertise_matches[: request.top_k]
    ]
    return payload


def _ondemand_search(request: SearchRequest) -> dict:
    """Live discovery per request, then the unchanged RAG pipeline."""
    from faculty_radar.rag.pipeline import RagPipeline

    discovery = getattr(app.state, "discovery", None)
    if discovery is None:
        raise HTTPException(status_code=503, detail="discovery not ready")
    found = discovery.discover(request.query.strip())
    if not found.profiles or not found.documents:
        # Genuinely nothing in scope supports this query: insufficient, zero
        # fabricated results. Same contract as corpus mode.
        return {
            "query": request.query.strip(),
            "status": "insufficient_evidence",
            "verified": False,
            "answer_text": (
                "insufficient_evidence: no consented faculty evidence "
                "supports an answer to this query."
            ),
            "warnings": ["no in-scope evidence discovered"],
            "llm_model": "extractive-v1",
            "claims": [],
            "matches": [],
            "corpus_source": "live-openalex",
            "discovery": {
                "provider": found.provider,
                "cached": found.cached,
                "works_retrieved": found.works_retrieved,
                "api_calls": found.api_calls,
                "scope_institutions": discovery.scope_ids,
                "crossref_checked": found.crossref_checked,
                "crossref_verified": found.crossref_verified,
                "crossref_enriched": found.crossref_enriched,
                "crossref_mismatched": found.crossref_mismatched,
                "query": request.query.strip(),
            },
        }

    settings = discovery.settings
    pipeline = RagPipeline(list(found.profiles), list(found.works), list(found.documents), settings)
    profiles_by_id = {p.faculty_id: p for p in found.profiles}
    answer = pipeline.answer(request.query.strip(), top_k=request.top_k)
    payload = _answer_payload(answer)
    payload["corpus_source"] = "live-openalex"
    payload["corpus_profiles"] = len(found.profiles)
    payload["corpus_documents"] = len(found.documents)
    payload["matches"] = [
        _match_payload(match, profiles_by_id) for match in answer.expertise_matches[: request.top_k]
    ]
    payload["discovery"] = {
        "provider": found.provider,
        "cached": found.cached,
        "works_retrieved": found.works_retrieved,
        "api_calls": found.api_calls,
        "scope_institutions": discovery.scope_ids,
        "crossref_checked": found.crossref_checked,
        "crossref_verified": found.crossref_verified,
        "crossref_enriched": found.crossref_enriched,
        "crossref_mismatched": found.crossref_mismatched,
        "query": request.query.strip(),
    }
    return payload
