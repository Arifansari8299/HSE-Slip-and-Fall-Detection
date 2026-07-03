"""
GeminiRAGAgent — Production RAG pipeline for HSE Incident Q&A

Architecture:
  alerts/logs.csv + alerts/running_logs.csv
        │
        ▼
  Document Loader  (structured natural language chunks)
        │
        ▼
  Gemini text-embedding-004  →  ChromaDB (local, persistent)
        │                              │
        │         ┌────────────────────┘
        │         │  cosine similarity retrieval (top-K)
        ▼         ▼
  Retrieved context chunks
        │
        ▼
  Gemini gemini-2.0-flash-lite  (grounded generation)
  System prompt enforces: only answer from context,
  never hallucinate, cite timestamps and IDs.
        │
        ▼
  Factual natural language answer

Setup:
    pip install google-genai chromadb

Usage:
    python src/gemini_rag_agent.py
    or: from src.gemini_rag_agent import GeminiRAGAgent; agent = GeminiRAGAgent(api_key="...")
"""

import csv
import os
import sys
import datetime
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
#  SYSTEM PROMPT — zero hallucination contract
# ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an HSE (Health, Safety & Environment) Incident Intelligence Assistant.

Your ONLY job is to answer questions about workplace safety incidents based on the 
provided context retrieved from the incident database.

STRICT RULES — you MUST follow these without exception:
1. ONLY use facts present in the provided context. Never invent data.
2. If the context does not contain enough information, say exactly:
   "I don't have enough data in the incident records to answer this question."
3. Always cite specific timestamps, Person IDs, and event types when answering.
4. Never guess dates, counts, or person IDs.
5. When asked for counts, count only what is explicitly in the context.
6. Format numbers clearly. Use bullet points for lists.
7. Keep answers concise and professional.

Event types in the database:
- SLIP_FALL: Person fell or slipped. Logged in alerts/logs.csv.
- RUNNING_PANIC: Person running or showing panic behavior. Logged in alerts/running_logs.csv.

You are a factual reporting tool, not a creative assistant."""


# ─────────────────────────────────────────────────────────
#  DOCUMENT LOADERS
# ─────────────────────────────────────────────────────────

def _load_fall_docs(csv_path: str) -> list[dict]:
    """Load alerts/logs.csv → list of document dicts with text + metadata."""
    docs = []
    if not os.path.exists(csv_path):
        logger.warning("Fall CSV not found: %s", csv_path)
        return docs

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("timestamp"):
                continue
            try:
                # Support 4-col (old) and 5-col (new with event_type)
                if len(row) >= 5:
                    ts, event_type, track_id, aspect_ratio, screenshot = (
                        row[0].strip(), row[1].strip(), row[2].strip(),
                        row[3].strip(), row[4].strip()
                    )
                elif len(row) == 4:
                    ts, track_id, aspect_ratio, screenshot = (
                        row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                    )
                    event_type = "SLIP_FALL"
                else:
                    continue

                dt = datetime.datetime.fromisoformat(ts)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
                hour = dt.hour
                weekday = dt.strftime("%A")
                shift = (
                    "office hours" if 9 <= hour < 18 else
                    "evening shift" if 18 <= hour < 22 else
                    "night shift"
                )
                # Rich natural language chunk — this is what gets embedded
                text = (
                    f"Incident report: On {date_str} ({weekday}) at {time_str} "
                    f"during {shift}, a SLIP_FALL (fall detection) alert was triggered. "
                    f"Person Track ID: {track_id}. "
                    f"Aspect ratio value: {aspect_ratio}. "
                    f"Screenshot saved: {screenshot if screenshot else 'none'}. "
                    f"Source: alerts/logs.csv."
                )

                # Stable unique ID for ChromaDB deduplication
                doc_id = hashlib.md5(f"{ts}_{track_id}_fall".encode()).hexdigest()

                docs.append({
                    "id": doc_id,
                    "text": text,
                    "timestamp": ts,
                    "date": date_str,
                    "time": time_str,
                    "hour": str(hour),
                    "weekday": weekday,
                    "shift": shift,
                    "event_type": "SLIP_FALL",
                    "track_id": track_id,
                    "aspect_ratio": aspect_ratio,
                    "screenshot": screenshot,
                    "source": "alerts/logs.csv",
                })
            except (ValueError, IndexError):
                continue

    return docs


def _load_running_docs(csv_path: str) -> list[dict]:
    """Load alerts/running_logs.csv → list of document dicts."""
    docs = []
    if not os.path.exists(csv_path):
        logger.warning("Running CSV not found: %s", csv_path)
        return docs

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("timestamp"):
                continue
            try:
                ts = row[0].strip()
                track_id = row[1].strip()
                velocity = row[2].strip() if len(row) > 2 else "0"
                shift_raw = row[3].strip() if len(row) > 3 else ""

                dt = datetime.datetime.fromisoformat(ts)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
                hour = dt.hour
                weekday = dt.strftime("%A")

                shift = shift_raw.replace("_", " ") if shift_raw else (
                    "office hours" if 9 <= hour < 18 else
                    "evening shift" if 18 <= hour < 22 else
                    "night shift"
                )

                text = (
                    f"Incident report: On {date_str} ({weekday}) at {time_str} "
                    f"during {shift}, a RUNNING_PANIC (running or panic behavior) alert "
                    f"was triggered. Person Track ID: {track_id}. "
                    f"Detected velocity: {velocity}. "
                    f"Source: alerts/running_logs.csv."
                )

                doc_id = hashlib.md5(f"{ts}_{track_id}_running".encode()).hexdigest()

                docs.append({
                    "id": doc_id,
                    "text": text,
                    "timestamp": ts,
                    "date": date_str,
                    "time": time_str,
                    "hour": str(hour),
                    "weekday": weekday,
                    "shift": shift,
                    "event_type": "RUNNING_PANIC",
                    "track_id": track_id,
                    "aspect_ratio": velocity,
                    "screenshot": "",
                    "source": "alerts/running_logs.csv",
                })
            except (ValueError, IndexError):
                continue

    return docs


# ─────────────────────────────────────────────────────────
#  CHROMA STORE — persistent local vector database
# ─────────────────────────────────────────────────────────

class ChromaStore:
    """
    Manages a persistent ChromaDB collection.
    Uses Gemini text-embedding-004 to embed documents.
    Supports upsert so re-indexing is idempotent.
    """

    COLLECTION_NAME = "hse_incidents"
    EMBED_MODEL = "gemini-embedding-001"
    EMBED_BATCH_SIZE = 50  # Gemini allows up to 100 per batch call

    def __init__(self, gemini_client, persist_dir: str = "alerts/chroma_db"):
        import chromadb
        self._client_gemini = gemini_client
        self._chroma = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._chroma.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("[ChromaStore] Collection '%s' ready (%d docs)",
                    self.COLLECTION_NAME, self._collection.count())

    def count(self) -> int:
        return self._collection.count()

    def upsert(self, documents: list[dict]):
        """Embed and upsert documents in batches. Skips already-indexed IDs."""
        if not documents:
            return

        # Filter out already indexed
        existing_ids = set(self._collection.get(
            ids=[d["id"] for d in documents]
        )["ids"])
        new_docs = [d for d in documents if d["id"] not in existing_ids]

        if not new_docs:
            logger.info("[ChromaStore] All %d documents already indexed.", len(documents))
            return

        logger.info("[ChromaStore] Embedding %d new documents...", len(new_docs))

        for i in range(0, len(new_docs), self.EMBED_BATCH_SIZE):
            batch = new_docs[i: i + self.EMBED_BATCH_SIZE]
            texts = [d["text"] for d in batch]

            # Gemini batch embed
            response = self._client_gemini.models.embed_content(
                model=self.EMBED_MODEL,
                contents=texts,
            )
            embeddings = [e.values for e in response.embeddings]

            self._collection.upsert(
                ids=[d["id"] for d in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{
                    "timestamp": d["timestamp"],
                    "date": d["date"],
                    "time": d["time"],
                    "hour": d["hour"],
                    "weekday": d["weekday"],
                    "shift": d["shift"],
                    "event_type": d["event_type"],
                    "track_id": d["track_id"],
                    "aspect_ratio": d["aspect_ratio"],
                    "screenshot": d["screenshot"],
                    "source": d["source"],
                } for d in batch],
            )
            logger.info("[ChromaStore] Batch %d/%d indexed.",
                        min(i + self.EMBED_BATCH_SIZE, len(new_docs)), len(new_docs))

        logger.info("[ChromaStore] ✅ Upsert complete. Total: %d", self._collection.count())

    def query(self, query_text: str, top_k: int = 15) -> list[dict]:
        """
        Embed the query with Gemini and retrieve top-K similar documents.
        Returns list of dicts with 'text' and 'metadata'.
        """
        response = self._client_gemini.models.embed_content(
            model=self.EMBED_MODEL,
            contents=[query_text],
        )
        query_embedding = response.embeddings[0].values

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text": doc,
                "metadata": meta,
                "similarity": round(1 - dist, 4),  # cosine distance → similarity
            })

        return retrieved


# ─────────────────────────────────────────────────────────
#  GEMINI RAG AGENT — main interface
# ─────────────────────────────────────────────────────────

class GeminiRAGAgent:
    """
    Production RAG agent using:
      - Gemini text-embedding-004 for semantic embeddings
      - ChromaDB for local persistent vector storage
      - Gemini gemini-2.0-flash-lite for grounded answer generation

    Zero hallucination: LLM is instructed to ONLY use retrieved context.
    If context is insufficient, it says so explicitly.
    """

    GENERATION_MODEL = "gemini-2.0-flash"
    TOP_K = 15

    def __init__(
        self,
        api_key: str,
        fall_csv: str = "alerts/logs.csv",
        running_csv: str = "alerts/running_logs.csv",
        chroma_dir: str = "alerts/chroma_db",
        generation_model: str = "gemini-2.0-flash",
        groq_cfg: Optional[dict] = None,
    ):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            )

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._fall_csv = fall_csv
        self._running_csv = running_csv
        self.GENERATION_MODEL = generation_model
        self._groq_cfg = groq_cfg or {}

        # Initialize vector store
        self._store = ChromaStore(self._client, persist_dir=chroma_dir)

        # Index documents on startup
        self._index()

    def _index(self):
        """Load both CSVs and upsert into ChromaDB."""
        fall_docs = _load_fall_docs(self._fall_csv)
        running_docs = _load_running_docs(self._running_csv)
        all_docs = fall_docs + running_docs

        print(f"[GeminiRAGAgent] Loaded {len(fall_docs)} fall + "
              f"{len(running_docs)} running documents.")
        print(f"[GeminiRAGAgent] Indexing into ChromaDB (skips already-indexed)...")

        self._store.upsert(all_docs)
        print(f"[GeminiRAGAgent] ✅ Vector DB ready — {self._store.count()} total documents.")

    def reindex(self):
        """Call this after new events are appended to CSVs."""
        print("[GeminiRAGAgent] Re-indexing from CSVs...")
        self._index()

    def _generate_groq(self, prompt: str) -> str:
        """Fallback generation via Groq API using official groq SDK."""
        groq_key = self._groq_cfg.get("api_key", "")
        groq_model = self._groq_cfg.get("model", "llama-3.1-8b-instant")

        if not groq_key or groq_key == "your_groq_api_key_here":
            return (
                "Gemini access denied and no Groq fallback key configured.\n"
                "Get a free key at https://console.groq.com and set it in "
                "config/settings.yaml under groq.api_key"
            )

        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
            print(f"[GeminiRAGAgent] ⚡ Answered via Groq ({groq_model})")
            return answer
        except ImportError:
            return "groq package not installed. Run: pip install groq"
        except Exception as e:
            return f"Both Gemini and Groq failed. Groq error: {e}"
        """Fallback generation via Groq API (free, no credit card required)."""
        groq_key = self._groq_cfg.get("api_key", "")
        groq_model = self._groq_cfg.get("model", "llama-3.1-8b-instant")

        if not groq_key or groq_key == "your_groq_api_key_here":
            return (
                "Gemini access denied (403) and no Groq fallback key configured.\n"
                "Get a free Groq key at https://console.groq.com and set it in "
                "config/settings.yaml under groq.api_key"
            )

        try:
            import urllib.request, json as _json
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            }
            body = _json.dumps({
                "model": groq_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 1024,
            }).encode()

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
                answer = data["choices"][0]["message"]["content"].strip()
                print(f"[GeminiRAGAgent] ⚡ Answered via Groq fallback ({groq_model})")
                return answer
        except Exception as e:
            return f"Both Gemini and Groq failed. Groq error: {e}"

    def ask(self, question: str, verbose: bool = False) -> str:
        """
        Answer a natural language question about HSE incidents.

        Pipeline:
          1. Embed question with Gemini text-embedding-004
          2. Retrieve top-K semantically similar documents from ChromaDB
          3. Build grounded prompt with retrieved context
          4. Generate answer with Gemini (zero hallucination system prompt)

        Args:
            question: Natural language question
            verbose:  If True, prints retrieved context for debugging

        Returns:
            Factual answer string grounded in retrieved context.
        """
        if self._store.count() == 0:
            return ("No incident data indexed yet. "
                    "Make sure alerts/logs.csv exists and call reindex().")

        # Step 1: Retrieve semantically relevant documents
        retrieved = self._store.query(question, top_k=self.TOP_K)

        if not retrieved:
            return ("No relevant incident records found for your question. "
                    "Try rephrasing or ask about falls, running alerts, or specific dates.")

        if verbose:
            print(f"\n[Retrieved {len(retrieved)} documents]")
            for i, r in enumerate(retrieved[:3], 1):
                print(f"  [{i}] sim={r['similarity']} | {r['text'][:120]}...")

        # Step 2: Build grounded context block
        context_lines = []
        for i, r in enumerate(retrieved, 1):
            context_lines.append(
                f"[Record {i}] "
                f"Date: {r['metadata'].get('date', 'N/A')} | "
                f"Time: {r['metadata'].get('time', 'N/A')} | "
                f"Event: {r['metadata'].get('event_type', 'N/A')} | "
                f"Person ID: {r['metadata'].get('track_id', 'N/A')} | "
                f"Shift: {r['metadata'].get('shift', 'N/A')} | "
                f"Source: {r['metadata'].get('source', 'N/A')}"
            )
        context_block = "\n".join(context_lines)

        # Step 3: Build prompt — context first, then question
        prompt = f"""RETRIEVED INCIDENT RECORDS FROM DATABASE:
{context_block}

---
USER QUESTION: {question}

Answer the question using ONLY the records above. 
Count carefully if asked for numbers. 
If you cannot answer from the records above, say so clearly."""

        # Step 4: Try Gemini first, fall back to Groq on any auth/quota error
        try:
            response = self._client.models.generate_content(
                model=self.GENERATION_MODEL,
                contents=prompt,
                config=self._genai.types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["403", "429", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED"]):
                print(f"[GeminiRAGAgent] Gemini unavailable ({err[:60]}...) → switching to Groq")
                return self._generate_groq(prompt)
            raise


# ─────────────────────────────────────────────────────────
#  CLI — interactive Q&A terminal
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import yaml

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "settings.yaml"
    )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    API_KEY = cfg.get("gemini", {}).get("api_key", "")
    CHROMA_DIR = cfg.get("gemini", {}).get("chroma_dir", "alerts/chroma_db")
    GEN_MODEL = cfg.get("gemini", {}).get("generation_model", "gemini-1.5-flash")
    FALL_CSV = cfg.get("csv_log_path", "alerts/logs.csv")
    RUNNING_CSV = cfg.get("running_log_path", "alerts/running_logs.csv")

    if not API_KEY or API_KEY == "your_gemini_api_key_here":
        print("ERROR: Set your Gemini API key in config/settings.yaml under gemini.api_key")
        print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  HSE GEMINI RAG AGENT — Incident Intelligence Q&A")
    print("  Powered by: Gemini Embeddings + ChromaDB + Gemini LLM")
    print("=" * 60)

    GROQ_CFG = cfg.get("groq", {})

    agent = GeminiRAGAgent(
        api_key=API_KEY,
        fall_csv=FALL_CSV,
        running_csv=RUNNING_CSV,
        chroma_dir=CHROMA_DIR,
        generation_model=GEN_MODEL,
        groq_cfg=GROQ_CFG,
    )

    EXAMPLES = [
        "How many fall incidents happened this week?",
        "Who had the most incidents overall?",
        "When was the last running alert detected?",
        "Give me a summary of all incidents on 2026-06-18",
        "How many incidents occurred during night shift?",
        "Which person ID triggered the most fall alerts?",
        "What happened on June 25?",
        "List the most recent 5 incidents",
    ]

    print("\nExample questions:")
    for q in EXAMPLES:
        print(f"  • {q}")
    print("\nType 'reindex' to reload CSVs, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if user_input.lower() == "reindex":
            agent.reindex()
            continue

        print("\nAgent: ", end="", flush=True)
        try:
            answer = agent.ask(user_input)
            print(answer)
        except Exception as e:
            print(f"[ERROR] {e}")
        print()
