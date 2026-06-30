"""
RAGAgent — Retrieval-Augmented Generation for HSE Incident Q&A

Architecture (no LLM, no external API, no Ollama):

  alerts/logs.csv  ──►  Document Loader
                             │
                             ▼
                      TF-IDF Vectorizer   ← lightweight, pure sklearn
                             │
                             ▼
                      Vector Index (in-memory)
                             │
                   ┌─────────┴──────────┐
                   │    RAG Pipeline     │
                   │  1. Retrieve top-K  │  ← cosine similarity search
                   │  2. Parse context   │  ← structured CSV rows
                   │  3. Answer engine   │  ← rule-based NLU + stats
                   └─────────┬──────────┘
                             │
                             ▼
                    Natural language answer

Usage:
    python src/rag_agent.py
    or import and call: agent.ask("How many falls today?")

Requirements:
    pip install scikit-learn
"""

import csv
import os
import re
import datetime
from collections import defaultdict, Counter

# scikit-learn is already available in most ML environments
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ─────────────────────────────────────────────────────────
#  DOCUMENT LOADER — reads CSV into structured documents
# ─────────────────────────────────────────────────────────

def _load_documents(csv_path: str) -> list[dict]:
    """Loads fall detection CSV (alerts/logs.csv) into structured documents."""
    documents = []
    if not os.path.exists(csv_path):
        return documents

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row or row[0].startswith("timestamp"):
                continue  # skip header

            # Support both old format (4 cols) and new format (5 cols with event_type)
            try:
                if len(row) >= 5:
                    timestamp, event_type, track_id, aspect_ratio, screenshot = (
                        row[0].strip(), row[1].strip(), row[2].strip(),
                        row[3].strip(), row[4].strip()
                    )
                elif len(row) == 4:
                    timestamp, track_id, aspect_ratio, screenshot = (
                        row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                    )
                    # Infer event type from screenshot filename
                    event_type = "RUNNING_PANIC" if "running" in screenshot.lower() else "SLIP_FALL"
                else:
                    continue

                # Parse timestamp
                dt = datetime.datetime.fromisoformat(timestamp)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
                hour = dt.hour
                weekday = dt.strftime("%A")

                shift = (
                    "office hours" if 9 <= hour < 18 else
                    "evening shift" if 18 <= hour < 22 else
                    "night shift"
                )

                event_label = (
                    "fall detection" if event_type == "SLIP_FALL"
                    else "running or panic detection"
                )

                # Natural language text for TF-IDF indexing
                text = (
                    f"On {date_str} {weekday} at {time_str} during {shift}, "
                    f"a {event_label} alert was triggered for person with track ID {track_id}. "
                    f"Event type: {event_type}. "
                    f"Aspect ratio: {aspect_ratio}. "
                    f"Screenshot: {screenshot}."
                )

                documents.append({
                    "text": text,
                    "timestamp": timestamp,
                    "date": date_str,
                    "time": time_str,
                    "hour": hour,
                    "weekday": weekday,
                    "shift": shift,
                    "event_type": event_type,
                    "track_id": track_id,
                    "aspect_ratio": aspect_ratio,
                    "screenshot": screenshot,
                })

            except (ValueError, IndexError):
                continue

    return documents


def _load_running_documents(csv_path: str) -> list[dict]:
    """Loads running detection CSV (alerts/running_logs.csv) into structured documents."""
    documents = []
    if not os.path.exists(csv_path):
        return documents

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("timestamp"):
                continue
            try:
                timestamp = row[0].strip()
                track_id = row[1].strip()
                velocity = row[2].strip() if len(row) > 2 else "0"
                shift = row[3].strip() if len(row) > 3 else ""

                dt = datetime.datetime.fromisoformat(timestamp)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
                hour = dt.hour
                weekday = dt.strftime("%A")

                if not shift:
                    shift = (
                        "office hours" if 9 <= hour < 18 else
                        "evening shift" if 18 <= hour < 22 else
                        "night shift"
                    )
                else:
                    shift = shift.replace("_", " ")

                text = (
                    f"On {date_str} {weekday} at {time_str} during {shift}, "
                    f"a running or panic detection alert was triggered for person with track ID {track_id}. "
                    f"Event type: RUNNING_PANIC. Velocity: {velocity}."
                )

                documents.append({
                    "text": text,
                    "timestamp": timestamp,
                    "date": date_str,
                    "time": time_str,
                    "hour": hour,
                    "weekday": weekday,
                    "shift": shift,
                    "event_type": "RUNNING_PANIC",
                    "track_id": track_id,
                    "aspect_ratio": velocity,
                    "screenshot": "",
                })
            except (ValueError, IndexError):
                continue

    return documents


# ─────────────────────────────────────────────────────────
#  VECTOR INDEX — TF-IDF based retrieval
# ─────────────────────────────────────────────────────────

class VectorIndex:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self._vectorizer = None
        self._matrix = None
        if _SKLEARN_AVAILABLE and documents:
            self._build()

    def _build(self):
        texts = [d["text"] for d in self.documents]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Returns top-K most relevant documents for the query."""
        if not _SKLEARN_AVAILABLE or self._vectorizer is None:
            return self.documents[:top_k]
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


# ─────────────────────────────────────────────────────────
#  ANSWER ENGINE — rule-based NLU over retrieved context
# ─────────────────────────────────────────────────────────

class AnswerEngine:
    """
    Parses user intent from the query and computes answers
    from retrieved documents using aggregation and filtering.
    """

    def answer(self, query: str, docs: list[dict], all_docs: list[dict]) -> str:
        q = query.lower()

        # ── Dispatch by intent ──
        if self._match(q, ["how many", "count", "total", "number of"]):
            return self._answer_count(q, docs, all_docs)

        if self._match(q, ["when", "last time", "most recent", "latest"]):
            return self._answer_when(q, docs)

        if self._match(q, ["who", "which person", "person id", "track id"]):
            return self._answer_who(q, docs, all_docs)

        if self._match(q, ["peak", "busiest", "most", "highest", "worst"]):
            return self._answer_peak(q, docs, all_docs)

        if self._match(q, ["summary", "report", "overview", "today", "this week"]):
            return self._answer_summary(q, all_docs)

        if self._match(q, ["list", "show", "what events", "all incidents"]):
            return self._answer_list(q, docs)

        if self._match(q, ["shift", "night", "office", "evening"]):
            return self._answer_shift(q, docs, all_docs)

        # Default: summarize retrieved context
        if docs:
            return self._answer_summary_from_docs(docs)

        return "I couldn't find relevant incident records for your question."

    # ── Intent handlers ──

    def _answer_count(self, q: str, docs: list[dict], all_docs: list[dict]) -> str:
        filtered = self._filter_by_event(q, all_docs)
        filtered = self._filter_by_date(q, filtered)
        event_label = self._event_label(q)

        if not filtered:
            return f"No {event_label} events found matching your query."

        count = len(filtered)
        date_hint = self._date_hint(q)
        return (
            f"There are <b>{count}</b> {event_label} incident(s) "
            f"{date_hint}recorded in the system."
        )

    def _answer_when(self, q: str, docs: list[dict]) -> str:
        filtered = self._filter_by_event(q, docs)
        if not filtered:
            return "No matching events found."
        latest = max(filtered, key=lambda d: d["timestamp"])
        return (
            f"The most recent {self._event_label(q)} event was on "
            f"<b>{latest['date']}</b> at <b>{latest['time']}</b> "
            f"({latest['weekday']}, {latest['shift']}) for Person ID #{latest['track_id']}."
        )

    def _answer_who(self, q: str, docs: list[dict], all_docs: list[dict]) -> str:
        filtered = self._filter_by_event(q, all_docs)
        filtered = self._filter_by_date(q, filtered)
        if not filtered:
            return "No matching events found."
        counter = Counter(d["track_id"] for d in filtered)
        top = counter.most_common(5)
        lines = "\n".join(f"  Person ID #{pid}: {cnt} incident(s)" for pid, cnt in top)
        return (
            f"Top persons by {self._event_label(q)} incident count"
            f"{' ' + self._date_hint(q) if self._date_hint(q) else ''}:\n{lines}"
        )

    def _answer_peak(self, q: str, docs: list[dict], all_docs: list[dict]) -> str:
        filtered = self._filter_by_event(q, all_docs)
        if not filtered:
            return "No events found."

        if "day" in q or "date" in q:
            counter = Counter(d["date"] for d in filtered)
            peak, count = counter.most_common(1)[0]
            return f"The busiest day was <b>{peak}</b> with <b>{count}</b> incidents."

        if "hour" in q or "time" in q:
            counter = Counter(d["hour"] for d in filtered)
            peak, count = counter.most_common(1)[0]
            return f"The peak hour is <b>{peak:02d}:00</b> with <b>{count}</b> incidents."

        # Default: worst day
        counter = Counter(d["date"] for d in filtered)
        peak, count = counter.most_common(1)[0]
        return f"The worst day overall was <b>{peak}</b> with <b>{count}</b> {self._event_label(q)} incidents."

    def _answer_summary(self, q: str, all_docs: list[dict]) -> str:
        filtered = self._filter_by_date(q, all_docs)
        if not filtered:
            return "No incidents found for the requested period."

        total = len(filtered)
        fall_count = sum(1 for d in filtered if d["event_type"] == "SLIP_FALL")
        run_count = sum(1 for d in filtered if d["event_type"] == "RUNNING_PANIC")
        by_day = Counter(d["date"] for d in filtered)
        worst_day = by_day.most_common(1)[0] if by_day else ("N/A", 0)
        by_shift = Counter(d["shift"] for d in filtered)
        period = self._date_hint(q).strip() or "overall"

        return (
            f"HSE Incident Summary {period}:\n"
            f"  Total incidents   : {total}\n"
            f"  Fall detections   : {fall_count}\n"
            f"  Running/Panic     : {run_count}\n"
            f"  Worst day         : {worst_day[0]} ({worst_day[1]} incidents)\n"
            f"  By shift          : " +
            ", ".join(f"{s}: {c}" for s, c in by_shift.items())
        )

    def _answer_list(self, q: str, docs: list[dict]) -> str:
        filtered = self._filter_by_event(q, docs)[:10]
        if not filtered:
            return "No events found."
        lines = "\n".join(
            f"  [{d['date']} {d['time']}] {d['event_type']} — Person ID #{d['track_id']}"
            for d in sorted(filtered, key=lambda x: x["timestamp"], reverse=True)
        )
        return f"Recent incidents:\n{lines}"

    def _answer_shift(self, q: str, docs: list[dict], all_docs: list[dict]) -> str:
        shift_map = {
            "night": "night shift",
            "office": "office hours",
            "evening": "evening shift",
        }
        target = next((v for k, v in shift_map.items() if k in q), None)
        filtered = [d for d in all_docs if target and d["shift"] == target] if target else all_docs
        count = len(filtered)
        label = target or "all shifts"
        return f"There are <b>{count}</b> incidents during <b>{label}</b>."

    def _answer_summary_from_docs(self, docs: list[dict]) -> str:
        fall = sum(1 for d in docs if d["event_type"] == "SLIP_FALL")
        run = sum(1 for d in docs if d["event_type"] == "RUNNING_PANIC")
        dates = sorted(set(d["date"] for d in docs))
        return (
            f"Found {len(docs)} related incident(s) across {len(dates)} day(s). "
            f"Falls: {fall}, Running/Panic: {run}. "
            f"Date range: {dates[0]} to {dates[-1]}."
        )

    # ── Filters & helpers ──

    def _filter_by_event(self, q: str, docs: list[dict]) -> list[dict]:
        if any(w in q for w in ["fall", "slip", "fallen", "medical"]):
            return [d for d in docs if d["event_type"] == "SLIP_FALL"]
        if any(w in q for w in ["run", "running", "panic", "evacuation"]):
            return [d for d in docs if d["event_type"] == "RUNNING_PANIC"]
        return docs

    def _filter_by_date(self, q: str, docs: list[dict]) -> list[dict]:
        today = datetime.date.today()
        if "today" in q:
            target = today.isoformat()
            return [d for d in docs if d["date"] == target]
        if "yesterday" in q:
            target = (today - datetime.timedelta(days=1)).isoformat()
            return [d for d in docs if d["date"] == target]
        if "this week" in q or "week" in q:
            cutoff = (today - datetime.timedelta(days=7)).isoformat()
            return [d for d in docs if d["date"] >= cutoff]
        if "this month" in q or "month" in q:
            cutoff = today.replace(day=1).isoformat()
            return [d for d in docs if d["date"] >= cutoff]
        # Try to extract a date like 2026-06-18
        m = re.search(r"\d{4}-\d{2}-\d{2}", q)
        if m:
            return [d for d in docs if d["date"] == m.group()]
        return docs

    def _event_label(self, q: str) -> str:
        if any(w in q for w in ["fall", "slip", "fallen"]):
            return "fall"
        if any(w in q for w in ["run", "running", "panic"]):
            return "running/panic"
        return "total"

    def _date_hint(self, q: str) -> str:
        if "today" in q:
            return "today "
        if "yesterday" in q:
            return "yesterday "
        if "this week" in q or "week" in q:
            return "this week "
        if "this month" in q or "month" in q:
            return "this month "
        m = re.search(r"\d{4}-\d{2}-\d{2}", q)
        if m:
            return f"on {m.group()} "
        return ""

    @staticmethod
    def _match(q: str, keywords: list[str]) -> bool:
        return any(k in q for k in keywords)


# ─────────────────────────────────────────────────────────
#  RAG AGENT — public interface
# ─────────────────────────────────────────────────────────

class RAGAgent:
    """
    RAG-based HSE Incident Q&A Agent.
    Reads from both alerts/logs.csv (falls) and alerts/running_logs.csv (running)
    and answers natural language questions about all incidents.
    """

    def __init__(self, csv_path: str = "alerts/logs.csv",
                 running_csv_path: str = "alerts/running_logs.csv"):
        self.csv_path = csv_path
        self.running_csv_path = running_csv_path
        self._documents: list[dict] = []
        self._index: VectorIndex | None = None
        self._engine = AnswerEngine()
        self._load()

    def _load(self):
        """Load and index documents from both CSVs."""
        fall_docs = _load_documents(self.csv_path)
        running_docs = _load_running_documents(self.running_csv_path)
        self._documents = fall_docs + running_docs
        self._index = VectorIndex(self._documents)
        print(
            f"[RAGAgent] ✅ Indexed {len(fall_docs)} fall + "
            f"{len(running_docs)} running records "
            f"({len(self._documents)} total)"
        )

    def reload(self):
        """Re-index after new events are appended to CSV."""
        self._load()

    def ask(self, question: str) -> str:
        """
        Answer a natural language question about HSE incidents.

        Args:
            question: e.g. "How many falls happened today?"

        Returns:
            Natural language answer string.
        """
        if not self._documents:
            return "No incident data found. Make sure alerts/logs.csv exists and has records."

        # Retrieve top relevant documents
        relevant_docs = self._index.retrieve(question, top_k=20)

        # Generate answer using retrieved context
        return self._engine.answer(question, relevant_docs, self._documents)


# ─────────────────────────────────────────────────────────
#  CLI — run interactively
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    csv_path = "alerts/logs.csv"
    running_csv_path = "alerts/running_logs.csv"
    agent = RAGAgent(csv_path=csv_path, running_csv_path=running_csv_path)

    print("\n" + "="*55)
    print("  HSE RAG AGENT — Incident Intelligence Q&A")
    print("  Type your question or 'quit' to exit")
    print("="*55)

    EXAMPLE_QUESTIONS = [
        "How many falls happened today?",
        "How many total incidents this week?",
        "Who had the most incidents?",
        "When was the last running alert?",
        "Give me a summary of all incidents",
        "Which day had the most incidents?",
        "How many incidents happened during night shift?",
    ]
    print("\nExample questions:")
    for q in EXAMPLE_QUESTIONS:
        print(f"  • {q}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting RAG Agent.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if user_input.lower() == "reload":
            agent.reload()
            print("[RAGAgent] Index reloaded.")
            continue

        answer = agent.ask(user_input)
        print(f"\nAgent: {answer}\n")
