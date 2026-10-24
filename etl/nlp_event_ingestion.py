#!/usr/bin/env python3
"""
SupplyPulse: Real-Time External Event Ingestion & NLP Parsing Pipeline
======================================================================
Enterprise-grade NLP extraction pipeline for supply chain disruption early-warnings.

Fetches live unstructured disaster bulletins (e.g., GDACS RSS feed) or realistic
synthetic alerts, applies Natural Language Processing (spaCy NER & rule-based scoring),
and ingests structured disruption records into the PostgreSQL `external_disruptions` table.

Author: SupplyPulse Team (Pradymna Adhari, Omkar Akhade, Vishal Amrutkar, Pravin Bolshette)
Review: Prof. Vanita Babbane (VIT Pune) | Fujitsu Data & AI Portfolio
"""

import os
import sys
import re
import hashlib
import logging
import argparse
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, IntegrityError

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SupplyPulse.NLP")

# ---------------------------------------------------------------------------
# Optional / Dynamic Library Imports (spaCy & feedparser)
# ---------------------------------------------------------------------------
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("Package 'feedparser' not found. Using urllib XML fallback.")

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        logger.info("Loaded spaCy model 'en_core_web_sm'.")
    except Exception as e:
        logger.warning(f"spaCy installed but model 'en_core_web_sm' not loaded: {e}. Fallback to rule-based NER.")
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    logger.warning("spaCy not installed. Fallback to rule-based NER.")


# ---------------------------------------------------------------------------
# Constants & Domain Mappings
# ---------------------------------------------------------------------------
DEFAULT_GDACS_FEED_URL = "https://www.gdacs.org/xml/rss.xml"
DEFAULT_DB_URL = "postgresql://admin:secret@localhost:5432/supplypulse"

# Supply Chain Maritime & Manufacturing Hub Gazetteer for Region Normalization
REGION_GAZETTEER = {
    "shanghai": "East China Sea / Yangtze Delta",
    "yangtze": "East China Sea / Yangtze Delta",
    "ningbo": "East China Sea / Ningbo Hub",
    "shenzhen": "South China Sea / Pearl River Delta",
    "guangzhou": "South China Sea / Pearl River Delta",
    "taiwan": "Taiwan Strait / Hsinchu Corridor",
    "hsinchu": "Taiwan Strait / Hsinchu Corridor",
    "taipei": "Taiwan Strait / Hsinchu Corridor",
    "hamburg": "Elbe Estuary & Northern Germany",
    "elbe": "Elbe Estuary & Northern Germany",
    "rotterdam": "Rhine-Meuse-Scheldt Delta",
    "rhine": "Rhine-Meuse-Scheldt Delta",
    "antwerp": "Scheldt Estuary / Antwerp Hub",
    "chennai": "Coromandel Coast / Bay of Bengal",
    "bengaluru": "Southern India Tech Corridor",
    "singapore": "Strait of Malacca / Jurong",
    "malacca": "Strait of Malacca",
    "busan": "Korea Strait / Busan Maritime Hub",
    "tokyo": "Tokyo Bay & Kanto Coastal Plain",
    "yokohama": "Tokyo Bay & Kanto Coastal Plain",
    "baltic": "Baltic Sea Shipping Approaches",
    "suez": "Suez Canal Maritime Chokepoint",
    "panama": "Panama Canal Transit Zone",
}

# Unstructured Realistic Disaster Bulletins (Simulation & Offline Fallback)
SAMPLE_UNSTRUCTURED_BULLETINS = [
    {
        "title": "Severe Category 4 Typhoon Approaching East China Coast",
        "description": "Severe Category 4 Typhoon approaching Shanghai port terminals with heavy flooding and torrential rain causing major maritime blockages across the Yangtze River Delta.",
        "pub_date": "2026-10-21",
    },
    {
        "title": "Magnitude 7.4 Earthquake Strikes Near Hsinchu Offshore Zone",
        "description": "Catastrophic Magnitude 7.4 Earthquake strikes near Hsinchu offshore zone and Taiwan Strait, triggering tsunami alarms and extensive halts to high-tech semiconductor foundries.",
        "pub_date": "2026-10-20",
    },
    {
        "title": "Massive Monsoon Surge Floods Industrial Parks Across Chennai",
        "description": "Massive monsoon flash flooding submerges automotive assembly hubs along Chennai and the Coromandel Coast with critical road closures and port export stoppages.",
        "pub_date": "2026-10-21",
    },
    {
        "title": "Dock Worker Strike Announced at Port of Hamburg",
        "description": "Indefinite port worker and dock labor strike announced across Hamburg container terminals on the Elbe River, halting all container crane offloading.",
        "pub_date": "2026-10-19",
    },
    {
        "title": "Rhine River Low Water Crisis Shuts Rotterdam Barge Convoys",
        "description": "Severe draught and falling river depths along the Rhine River near Rotterdam halt chemical barge shipments and container logistics into central Europe.",
        "pub_date": "2026-10-18",
    },
    {
        "title": "Container Feeder Vessel Grounding Blocks Strait of Malacca",
        "description": "A 14,000 TEU container vessel grounding causes severe navigation bottleneck and anchorage queue in the Strait of Malacca approaching Singapore Port.",
        "pub_date": "2026-10-21",
    },
    {
        "title": "Extreme Winter Blizzard Freezes Baltic Sea Shipping Corridors",
        "description": "Major subzero winter blizzard and rapid ice formation freeze commercial shipping approaches across the Baltic Sea, delaying roll-on/roll-off automotive carriers.",
        "pub_date": "2026-10-17",
    },
    {
        "title": "Berth Fire and Hazardous Material Explosion at Jurong Port",
        "description": "Emergency response teams battling container berth chemical fire at Jurong Port in Singapore; maritime authorities declare 5-kilometer safety exclusion zone.",
        "pub_date": "2026-10-21",
    },
    {
        "title": "Geopolitical Naval Drills Prompt Air and Sea Exclusion in Taiwan Strait",
        "description": "Sudden military live-fire drills and missile testing create an emergency exclusion corridor across the Taiwan Strait, rerouting commercial shipping.",
        "pub_date": "2026-10-20",
    },
    {
        "title": "Dense Fog Navigational Ban Imposed at Port of Ningbo-Zhoushan",
        "description": "Zero-visibility dense coastal fog triggers complete vessel movement suspension and anchorage delays outside Ningbo-Zhoushan container terminals.",
        "pub_date": "2026-10-19",
    },
]


# ---------------------------------------------------------------------------
# Module A: Feed Fetcher / Source Simulation
# ---------------------------------------------------------------------------
def fetch_raw_disaster_alerts(
    feed_url: str = DEFAULT_GDACS_FEED_URL,
    max_items: int = 15,
    simulate: bool = False,
) -> List[Dict[str, Any]]:
    """
    Pulls live unstructured disaster bulletins from GDACS XML/RSS feed,
    or falls back seamlessly to realistic simulated text bulletins if network
    is unavailable or simulation mode is requested.
    """
    if simulate:
        logger.info(f"Simulation mode active: Loading {len(SAMPLE_UNSTRUCTURED_BULLETINS)} realistic disaster bulletins.")
        return SAMPLE_UNSTRUCTURED_BULLETINS[:max_items]

    # Attempt Live Fetch
    logger.info(f"Attempting live fetch from GDACS feed: {feed_url}")
    try:
        if FEEDPARSER_AVAILABLE:
            feed = feedparser.parse(feed_url)
            if feed and hasattr(feed, "entries") and len(feed.entries) > 0:
                logger.info(f"Successfully retrieved {len(feed.entries)} live feed items from GDACS.")
                alerts = []
                for entry in feed.entries[:max_items]:
                    pub_date = None
                    if "published" in entry:
                        pub_date = entry.published
                    elif "published_parsed" in entry and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
                    else:
                        pub_date = str(date.today())

                    alerts.append({
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", "") or entry.get("description", ""),
                        "pub_date": pub_date,
                        "link": entry.get("link", ""),
                        "guid": entry.get("id", entry.get("link", "")),
                    })
                return alerts
            else:
                logger.warning("GDACS feed returned empty entries. Falling back to simulated bulletins.")
        else:
            # Fallback using urllib and xml.etree
            import urllib.request
            import xml.etree.ElementTree as ET

            req = urllib.request.Request(feed_url, headers={"User-Agent": "SupplyPulse-NLP/1.0"})
            with urllib.request.urlopen(req, timeout=8) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")
            logger.info(f"Retrieved {len(items)} items via XML parser fallback.")
            alerts = []
            for item in items[:max_items]:
                title = item.findtext("title") or ""
                desc = item.findtext("description") or ""
                pub_date = item.findtext("pubDate") or str(date.today())
                link = item.findtext("link") or ""
                alerts.append({
                    "title": title,
                    "description": desc,
                    "pub_date": pub_date,
                    "link": link,
                    "guid": link or title,
                })
            if alerts:
                return alerts

    except Exception as e:
        logger.warning(f"Could not fetch live feed ({e}). Gracefully falling back to simulated text bulletins.")

    # Return curated realistic bulletins on failure
    logger.info(f"Using {len(SAMPLE_UNSTRUCTURED_BULLETINS)} curated synthetic disruption bulletins.")
    return SAMPLE_UNSTRUCTURED_BULLETINS[:max_items]


# ---------------------------------------------------------------------------
# Module B: NLP Text Extraction Module (Keyword Rules + Gazetteer + spaCy NER)
# ---------------------------------------------------------------------------
# HONEST PIPELINE DESCRIPTION:
# This pipeline utilizes deterministic keyword heuristics combined with a curated
# maritime and trade hub place-name gazetteer. When the gazetteer misses, spaCy
# Named Entity Recognition (en_core_web_sm) is invoked as a secondary fallback.
# THIS IS NOT DEEP LEARNING. It is a deterministic rule-and-gazetteer extraction engine.
# ---------------------------------------------------------------------------

def classify_event_type(text_content: str) -> str:
    """
    Classify the disaster/disruption event type using keyword rules.
    """
    lower = text_content.lower()

    if any(k in lower for k in ["typhoon", "cyclone", "hurricane", "tropical storm", "tropical depression"]):
        return "Typhoon"
    elif any(k in lower for k in ["flood", "flooding", "flash flood", "submerged", "inundation", "torrential rain"]):
        return "Severe Flooding"
    elif any(k in lower for k in ["earthquake", "magnitude", "tremor", "seismic", "tsunami"]):
        return "Earthquake"
    elif any(k in lower for k in ["strike", "walkout", "labor dispute", "dock worker", "port worker"]):
        return "Port Worker Strike"
    elif any(k in lower for k in ["grounding", "grounded", "stoppage", "bottleneck", "navigational ban", "chokepoint", "canal"]):
        return "Shipping Bottleneck"
    elif any(k in lower for k in ["blizzard", "winter storm", "freezing ice", "subzero", "snowstorm"]):
        return "Winter Blizzard"
    elif any(k in lower for k in ["fire", "explosion", "chemical leak", "berth fire"]):
        return "Industrial Hazard"
    elif any(k in lower for k in ["geopolitical", "missile", "military drill", "exclusion zone", "sanctions", "blockade"]):
        return "Geopolitical Disruption"
    else:
        return "Severe Environmental Alert"


def extract_affected_region(text_content: str) -> str:
    """
    Extracts affected region using gazetteer first, then spaCy NER if gazetteer misses.
    """
    result = parse_headline(text_content)
    return result["region"]


def compute_severity_score(text_content: str) -> float:
    """
    Computes a normalized severity_score between 0.0 and 10.0 based on quantitative
    cues (Category, Magnitude, GDACS Alert Colors) and qualitative linguistic modifiers.
    """
    lower = text_content.lower()
    score = 5.0  # Base neutral severity

    # 1. Quantitative Indicators
    cat_match = re.search(r"category\s*([1-5])", lower)
    if cat_match:
        cat_num = int(cat_match.group(1))
        score = 4.8 + (cat_num * 1.0)

    mag_match = re.search(r"magnitude\s*([0-9]+(?:\.[0-9]+)?)", lower)
    if mag_match:
        mag = float(mag_match.group(1))
        score = min(10.0, max(4.0, (mag - 4.0) * 1.4 + 5.0))

    if "red alert" in lower or "red notification" in lower:
        score = max(score, 8.8)
    elif "orange alert" in lower or "orange notification" in lower:
        score = max(score, 6.8)
    elif "green alert" in lower or "green notification" in lower:
        score = min(score, 4.5)

    # 2. Qualitative Modifiers
    if any(k in lower for k in ["catastrophic", "devastating", "state of emergency", "deadly", "highest alert"]):
        score += 2.0
    elif any(k in lower for k in ["severe", "major", "torrential", "critical", "indefinite", "submerged"]):
        score += 1.2
    elif any(k in lower for k in ["moderate", "significant", "queue", "delays"]):
        score += 0.4
    elif any(k in lower for k in ["minor", "localized", "advisory", "watch", "safe"]):
        score -= 1.5

    return float(round(min(10.0, max(0.0, score)), 1))


def parse_headline(text_content: str) -> Dict[str, Any]:
    """
    Parses an unstructured supply chain disruption headline or text bulletin.

    Architecture:
    Combines keyword matching rules for event types with a place-name gazetteer
    for maritime and manufacturing hub identification. spaCy Named Entity Recognition
    (en_core_web_sm) is used as a fallback when the gazetteer misses.
    (Advisory heuristic NLP; NOT deep learning).

    Returns:
        dict with keys:
          - event_type: str
          - region: str
          - severity: float
          - entities: list[str]
          - method: str ('keyword_rules+gazetteer', 'keyword_rules+spacy_ner', or 'keyword_rules+heuristic_fallback')
    """
    clean_text = text_content.strip()
    lower = clean_text.lower()

    # 1. Event Type Classification
    event_type = classify_event_type(clean_text)

    # 2. Region & Entity Extraction
    entities: List[str] = []
    region: Optional[str] = None
    method = "keyword_rules+heuristic_fallback"

    # Step 2a: Direct Gazetteer Lookup
    for key, region_name in REGION_GAZETTEER.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            region = region_name
            entities.append(key.capitalize())
            method = "keyword_rules+gazetteer"
            break

    # Step 2b: spaCy NER Fallback if gazetteer misses
    if region is None:
        if SPACY_AVAILABLE and nlp is not None:
            doc = nlp(clean_text)
            ner_ents = [
                e.text.strip()
                for e in doc.ents
                if e.label_ in ("GPE", "LOC", "FAC") and len(e.text.strip()) > 2
            ]
            entities.extend(ner_ents)
            if ner_ents:
                for ent in ner_ents:
                    ent_low = ent.lower()
                    for key, region_name in REGION_GAZETTEER.items():
                        if key in ent_low:
                            region = region_name
                            method = "keyword_rules+spacy_ner"
                            break
                    if region:
                        break

                if not region:
                    region = ner_ents[0]
                    method = "keyword_rules+spacy_ner"

    # Step 2c: Rule-based regex fallback
    if region is None:
        match = re.search(r"\b(?:in|near|at|off|outside|approaching)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", clean_text)
        if match:
            cand = match.group(1).strip()
            region = cand
            entities.append(cand)
            method = "keyword_rules+heuristic_fallback"
        else:
            region = "Global Maritime Shipping Corridor"
            method = "keyword_rules+default_corridor"

    # 3. Severity Score
    severity = compute_severity_score(clean_text)

    # Deduplicate entities preserving order
    dedup_entities = list(dict.fromkeys(entities))

    return {
        "event_type": event_type,
        "region": region,
        "severity": severity,
        "entities": dedup_entities,
        "method": method,
    }


def parse_disruption_text(
    alert_raw: Dict[str, Any],
    index: int = 1,
) -> Dict[str, Any]:
    """
    Transforms raw unstructured alert dictionaries into structured records.
    """
    title = alert_raw.get("title", "")
    description = alert_raw.get("description", "")
    full_text = f"{title}. {description}".strip()

    parsed = parse_headline(full_text)

    # Parse Event Date
    raw_date = alert_raw.get("pub_date")
    parsed_date = date.today()
    if raw_date:
        try:
            if len(raw_date) >= 10 and raw_date[:10].count("-") == 2:
                parsed_date = datetime.strptime(raw_date[:10], "%Y-%m-%d").date()
            else:
                parsed_dt = pd.to_datetime(raw_date, errors="coerce")
                if not pd.isna(parsed_dt):
                    parsed_date = parsed_dt.date()
        except Exception:
            parsed_date = date.today()

    # Deterministic Event ID
    unique_string = alert_raw.get("guid") or f"{parsed['event_type']}_{parsed['region']}_{parsed_date}_{title[:30]}"
    hash_digest = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()[:8].upper()
    event_id = f"EVT-NLP-{hash_digest}"

    return {
        "event_id": event_id,
        "event_type": parsed["event_type"],
        "affected_region": parsed["region"],
        "severity_score": parsed["severity"],
        "event_date": parsed_date,
        "source_text": full_text[:600],
        "source": alert_raw.get("source", "GDACS / Reuters / Maritime Feed"),
        "parsed_method": parsed["method"],
        "raw_snippet": title[:90],
    }


# ---------------------------------------------------------------------------
# Module C: Database Ingestion Pipeline
# ---------------------------------------------------------------------------
def ensure_external_disruptions_table(engine, is_sqlite: bool = False):
    """
    Ensures external_disruptions table exists with source_text, source, and parsed_method.
    """
    with engine.begin() as conn:
        if is_sqlite:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS external_disruptions (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    affected_region TEXT NOT NULL,
                    severity_score REAL CHECK (severity_score >= 0.0 AND severity_score <= 10.0),
                    event_date TEXT DEFAULT CURRENT_DATE,
                    source_text TEXT,
                    source TEXT,
                    parsed_method TEXT
                );
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS external_disruptions (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    affected_region TEXT NOT NULL,
                    severity_score FLOAT CHECK (severity_score >= 0.0 AND severity_score <= 10.0),
                    event_date DATE DEFAULT CURRENT_DATE,
                    source_text TEXT,
                    source TEXT,
                    parsed_method TEXT
                );
            """))
            for col in ["source_text TEXT", "source TEXT", "parsed_method TEXT"]:
                try:
                    conn.execute(text(f"ALTER TABLE external_disruptions ADD COLUMN IF NOT EXISTS {col};"))
                except Exception:
                    pass


def ingest_disruptions_to_db(
    engine,
    records: List[Dict[str, Any]],
    is_sqlite: bool = False,
) -> Tuple[int, int]:
    """
    Inserts structured NLP disruption records into external_disruptions.
    """
    if not records:
        logger.warning("No records provided for ingestion.")
        return 0, 0

    ensure_external_disruptions_table(engine, is_sqlite=is_sqlite)

    if is_sqlite:
        insert_sql = """
            INSERT OR IGNORE INTO external_disruptions (
                event_id, event_type, affected_region, severity_score, event_date,
                source_text, source, parsed_method
            ) VALUES (
                :event_id, :event_type, :affected_region, :severity_score, :event_date,
                :source_text, :source, :parsed_method
            );
        """
    else:
        insert_sql = """
            INSERT INTO external_disruptions (
                event_id, event_type, affected_region, severity_score, event_date,
                source_text, source, parsed_method
            ) VALUES (
                :event_id, :event_type, :affected_region, :severity_score, :event_date,
                :source_text, :source, :parsed_method
            ) ON CONFLICT (event_id) DO NOTHING;
        """

    inserted_count = 0
    skipped_count = 0

    with engine.begin() as conn:
        for rec in records:
            params = {
                "event_id": rec["event_id"],
                "event_type": rec["event_type"],
                "affected_region": rec["affected_region"],
                "severity_score": rec["severity_score"],
                "event_date": str(rec["event_date"]),
                "source_text": rec.get("source_text", rec.get("raw_snippet", "")),
                "source": rec.get("source", "GDACS / Reuters / Maritime Feed"),
                "parsed_method": rec.get("parsed_method", "keyword_rules+gazetteer"),
            }
            res = conn.execute(text(insert_sql), params)
            if res.rowcount and res.rowcount > 0:
                inserted_count += 1
            else:
                skipped_count += 1

    return inserted_count, skipped_count



# ---------------------------------------------------------------------------
# CLI Argument Parser & Orchestration
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SupplyPulse: Real-Time External Event Ingestion & NLP Parsing Pipeline"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Attempt live GDACS RSS feed fetch (default: True with automatic fallback)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Force simulation mode using realistic predefined disaster bulletins",
    )
    parser.add_argument(
        "--feed-url",
        type=str,
        default=DEFAULT_GDACS_FEED_URL,
        help="Custom RSS/XML feed URL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Maximum number of alerts to process (default: 12)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and display extracted NLP attributes without writing to database",
    )
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="Ingest into local SQLite database (supplypulse.db) instead of PostgreSQL",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Custom PostgreSQL connection string (defaults to .env DATABASE_URL)",
    )
    return parser.parse_args()


def main():
    print("""
    =================================================================
       SUPPLYPULSE: AI SUPPLY CHAIN DISRUPTION PLATFORM
       NLP Real-Time External Event Ingestion & Parsing Pipeline
    =================================================================
    """)

    load_dotenv()
    args = parse_arguments()

    # Step 1: Fetch Raw Disasters Alerts (Live RSS or Simulated)
    logger.info("--- Step 1: Ingesting Raw Disaster Feeds ---")
    raw_alerts = fetch_raw_disaster_alerts(
        feed_url=args.feed_url,
        max_items=args.limit,
        simulate=args.simulate,
    )
    logger.info(f"Retrieved {len(raw_alerts)} raw alerts to parse.")

    # Step 2: NLP Information Extraction
    logger.info("--- Step 2: Executing NLP Attribute Extraction (spaCy + Heuristics) ---")
    parsed_records = []
    for idx, alert in enumerate(raw_alerts, 1):
        parsed = parse_disruption_text(alert, index=idx)
        parsed_records.append(parsed)

    df_results = pd.DataFrame(parsed_records)

    # Display Structured Results Table in Console
    print("\n" + "=" * 90)
    print(f" {'EVENT ID':<16} | {'EVENT TYPE':<22} | {'AFFECTED REGION':<32} | {'SEVERITY':<8} | {'DATE':<10}")
    print("=" * 90)
    for _, row in df_results.iterrows():
        print(f" {row['event_id']:<16} | {row['event_type']:<22} | {row['affected_region']:<32} | {row['severity_score']:>8.1f} | {str(row['event_date']):<10}")
    print("=" * 90 + "\n")

    # Step 3: Dry-Run Check
    if args.dry_run:
        logger.info("Dry-run mode active: Skipped database ingestion.")
        return 0

    # Step 4: Database Ingestion
    logger.info("--- Step 3: Database Ingestion Pipeline ---")
    if args.sqlite:
        db_url = "sqlite:///supplypulse.db"
        is_sqlite = True
        logger.info(f"Connecting to Local SQLite DB: {db_url}")
    else:
        is_sqlite = False
        db_url = (
            args.db_url
            or os.getenv("DATABASE_URL")
            or f"postgresql://{os.getenv('POSTGRES_USER', 'admin')}:{os.getenv('POSTGRES_PASSWORD', 'secret')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'supplypulse')}"
        )
        logger.info(f"Connecting to PostgreSQL DB: {db_url}")

    try:
        engine = create_engine(db_url, echo=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        logger.info("Successfully connected to database engine.")
    except OperationalError:
        logger.error(
            "\n" + "!" * 70 + "\n"
            "DATABASE CONNECTION FAILED!\n"
            f"Could not connect to PostgreSQL at: {db_url}\n\n"
            "Troubleshooting:\n"
            "  1. Start the PostgreSQL container:  docker compose up -d\n"
            "  2. Or test NLP pipeline offline with SQLite:  python etl/nlp_event_ingestion.py --sqlite\n"
            "  3. Or test with dry-run:  python etl/nlp_event_ingestion.py --dry-run\n"
            + "!" * 70
        )
        sys.exit(1)

    # Execute Idempotent Batch Ingestion
    inserted, skipped = ingest_disruptions_to_db(engine, parsed_records, is_sqlite=is_sqlite)
    logger.info(f"Ingestion Audit: {inserted} new records inserted, {skipped} duplicate events skipped (ON CONFLICT DO NOTHING).")

    # Verification: Query Total Count in DB
    with engine.connect() as conn:
        total_rows = conn.execute(text("SELECT COUNT(*) FROM external_disruptions;")).scalar()
        logger.info(f"Verification: Total records in 'external_disruptions' table: {total_rows}")

    logger.info("SUCCESS: SupplyPulse NLP Event Ingestion Pipeline Completed Successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
