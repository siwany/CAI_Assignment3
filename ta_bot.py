"""
CS 6604 / ECE 6256 — Assignment 3: The Course TA-Bot
Conversational RAG with Memory

Starter stub. Implement the methods below.
Do NOT rename the class or change the method signatures.
"""

from __future__ import annotations
import os
import sqlite3
import re
import requests

SYSTEM_PROMPT = (
    "You are the teaching assistant for CS 6604 / ECE 6256, a graduate course on "
    "conversational AI. Answer the student's question using the course paper excerpts "
    "provided with each question. Cite the papers you use inline by their ID in square "
    "brackets, e.g. [L04-04]. Only cite IDs that appear in the excerpts. If the excerpts "
    "do not contain the answer, say so briefly and give your best general answer without "
    "citations. Be accurate and concise: a few short paragraphs at most."
)

REWRITE_PROMPT = (
    "You rewrite a student's latest question into a standalone search query for a keyword "
    "(BM25) search over course papers. Use the conversation only to resolve pronouns and "
    "references like 'it', 'that approach', or 'these methods' into the concrete method, "
    "model, or paper names. If the student switches topics, ignore the earlier topic. "
    "Output only the query on one line: technical keywords and names, no explanation."
)

REFUSE_PATTERNS = [
    r"\b(diagnos\w*|medication|meds|prescri\w*|therap(y|ist)|antidepressant\w*|adhd|anxiety|depress\w*|bipolar|diabetes|pregnan\w*|disabilit\w*|mental health)\b",
    r"\b(salary|debt|loans?|income|credit (score|card)|bank account)\b",
    r"\b(religio\w*|ethnicity|citizenship|immigra\w*|visa status|undocumented|sexual orientation|gay|lesbian|bisexual|transgender)\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    r"[\w.+-]+@[\w-]+\.[\w.]+",
    r"\b\d{9}\b",
    r"\b(password|passcode|api key|home address|gpa)\b",
]
ALLOW_PATTERNS = [
    r"\bi(?:[’']m| am) (a|an)\b.*\b(phd|master'?s?|ms|undergrad\w*|ta|student|researcher)\b",
    r"\bi(?:[’']m| am) (taking|enrolled in|focusing on)\b",
    r"\bi(?:[’']m| am) (writing|working on|doing) (my|a|an|the)\b",   # project/assignment statements
    r"\bmy (project|thesis|research|dissertation|advisor)\b",
    r"\bi (work|research|study|focus) on\b",
    r"\b(my name is|call me|i go by)\b",
    r"\bi(?:[’']m| am) (most |really |especially )?interested in\b",
    r"\bi prefer\b",
    r"\bplease (always|keep|use|explain|cite)\b",
]

def _should_store(text: str) -> bool:
    """
    decide whether user message should be saved to memory across sessions based on REFUSE_PATTERNS and ALLOW_PATTERNS
    """
    lowered = text.lower()
    if any(re.search(p, lowered) for p in REFUSE_PATTERNS):
        return False
    return any(re.search(p, lowered) for p in ALLOW_PATTERNS)

def _extract_facts(text: str) -> list[str]:
    """
    split message into sentences and keep only the ones that pass _should_store
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if _should_store(s)]


class TABot:
    """Course TA-Bot with RAG retrieval and cross-session memory."""

    STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is", "are", "what", "how", "does", "do", "with", "by", "that", "this"}
    QUERY_MODES = ("current", "window", "rewrite")
    QUERY_MODE = "window"  ## default retrieval query mode; override via config["query_mode"] or TA_QUERY_MODE env var
    QUERY_WINDOW = 3 ## number of recent user turns used for retreival (ablate later)
    REWRITE_TURNS = 2      ## number of past (user, assistant) exchanges shown to the rewriter
    REWRITE_ANSWER_CHARS = 300  ## max characters of each past assistant answer shown to the rewriter
    REWRITE_MAX_CHARS = 300     ## rewrites longer than this are treated as failed
    TOP_K = 5        ## number of papers returned as citations
    HISTORY_TURNS = 4      ## number of past (user, assistant) exchanges sent to the LLM
    SNIPPET_CHARS = 1200   ## max characters of each retrieved chunk put in the prompt

    def __init__(self, config: dict) -> None:
        """
        Initialise the TA-Bot.

        Parameters
        ----------
        config : dict
            Required keys set by the harness (do not hard-code their values):
                endpoint   : str   – vLLM-compatible base URL
                model      : str   – model name served at the endpoint
                corpus_path: str   – path to the paper corpus directory
                temperature: float – generation temperature (0 for graded runs)
                seed       : int   – random seed for reproducibility
            You may add custom keys for your own use.
        """
        self.config = config
        # TODO: initialise your retrieval pipeline, memory store, etc.
        self.endpoint = config["endpoint"]
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.seed = config["seed"]
        self.timeout = config.get("timeout", 60)        ## seconds per LLM request
        self.max_tokens = config.get("max_tokens", 512)

        ## retrieval query mode (ablation axis): current message only / window concat / LLM rewrite
        self.query_mode = config.get("query_mode") or os.environ.get("TA_QUERY_MODE") or self.QUERY_MODE
        if self.query_mode not in self.QUERY_MODES:
            raise ValueError(f"query_mode must be one of {self.QUERY_MODES}, got {self.query_mode!r}")
        self.last_query = ""  ## most recent retrieval query, kept for ablation logging

        ## retreival pipeline: connect to corpus database
        db_path = os.path.join(config["corpus_path"], "corpus.db")
        self.db = sqlite3.connect(db_path, check_same_thread=False)

        ## memory store: user_id --> list of fact strings
        self.memory: dict[str, list[str]] = {}

        ## conversation state: session_id --> list of user turns (used for retrieval)
        self.sessions: dict[str, list[str]] = {}

        ## dialogue history: session_id --> list of {"role", "content"} messages (sent to the LLM)
        self.history: dict[str, list[dict]] = {}

    def chat(
        self,
        session_id: str,
        user_id: str,
        message: str,
    ) -> dict:
        """
        Process one user turn and return an answer with citations.

        Parameters
        ----------
        session_id : str
            Conversation scope — a new session_id starts a fresh
            conversation but keeps the user's cross-session memory.
        user_id : str
            Memory scope — identifies the user whose memory store
            to read from and write to.
        message : str
            The user's message.

        Returns
        -------
        dict
            {"answer": str, "citations": list[str]}
            - answer: the bot's response (complete, no streaming)
            - citations: list of paper_id strings from the corpus
              that were used to produce the answer. May be empty
              if no retrieval was needed.

        Raises
        ------
        TimeoutError
            You MUST raise TimeoutError if the endpoint is
            unreachable or a request exceeds your timeout.
            The harness catches it and marks this turn as 0.
            Do NOT let the process hang — raise, don't wait.
        """
        # TODO: implement retrieval, generation, and memory update
        query = self._build_query(session_id, message)
        self.last_query = query
        hits = self._search(query, k=self.TOP_K)
        citations = [pid for pid, _ in hits]

        ## records this user turn for future queries within same sessions
        self.sessions.setdefault(session_id, []).append(message)

        ## memory filter
        facts = self.memory.setdefault(user_id, [])
        for fact in _extract_facts(message):
            if fact not in facts:
                facts.append(fact)
                if len(facts) > 50:
                    del facts[0]

        ## generation
        messages = self._build_messages(session_id, user_id, message, hits)
        answer = self._generate(messages)

        ## records this exchange for future turns within same session
        self.history.setdefault(session_id, []).extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ])

        return {"answer": answer, "citations": citations}

    def reset_memory(self, user_id: str) -> None:
        """
        Clear all stored memory for the given user.

        The harness calls this before each memory-probe sequence
        to ensure a clean slate. After this call, ``get_memory(user_id)``
        must return an empty list.

        Parameters
        ----------
        user_id : str
            The user whose memory to clear.
        """
        # TODO: clear the memory store for this user
        self.memory[user_id] = []

    def get_memory(self, user_id: str) -> list[str]:
        """
        Return the current memory store for the given user as a
        list of human-readable fact strings.

        This is read-only — calling it must not change the store.
        The harness uses it to inspect what your bot chose to remember
        (and what it chose not to remember).

        Any internal representation is fine as long as each entry
        serialises to a readable string.

        Parameters
        ----------
        user_id : str
            The user whose memory to inspect.

        Returns
        -------
        list[str]
            Stored facts as readable strings. Empty list if nothing
            is stored (or after ``reset_memory``).
        """
        # TODO: return the contents of the memory store
        return list(self.memory.get(user_id, []))

    

    ### HELPER METHODS
    def _retrieve(self, query: str, k: int = 5) -> list[str]:
        """
        text query input
        returns up to k unique paper_ids that best match
        """
        return [pid for pid, _ in self._search(query, k)]

    def _search(self, query: str, k: int = 5) -> list[tuple[str, str]]:
        """
        BM25 search over chunks
        returns up to k (paper_id, chunk_id) pairs: unique papers, each with its best-scoring chunk
        """
        terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if t not in self.STOPWORDS]
        if not terms:
            return []
        match = " OR ".join(terms)
        rows = self.db.execute(
            "SELECT paper_id, chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, k * 10),
        ).fetchall()
        seen, hits = set(), []
        for pid, cid in rows:
            if pid not in seen:
                seen.add(pid)
                hits.append((pid, cid))
                if len(hits) == k:
                    break
        return hits

    def _build_messages(self, session_id: str, user_id: str, message: str,
                        hits: list[tuple[str, str]]) -> list[dict]:
        """
        assembles the chat prompt: system prompt + user memory, recent dialogue, then
        the retrieved excerpts and the current question
        """
        system = SYSTEM_PROMPT
        facts = self.memory.get(user_id, [])
        if facts:
            system += (
                "\n\nWhat this student told you in earlier sessions (their own words):\n"
                + "\n".join(f"- {f}" for f in facts)
                + "\nUse these to personalise your answer. If the student asks what you "
                "remember about them, restate these facts using their exact wording."
            )

        excerpts = []
        for pid, cid in hits:
            row = self.db.execute(
                "SELECT p.title, c.section, c.text FROM chunks c JOIN papers p USING(paper_id) "
                "WHERE c.chunk_id = ?",
                (cid,),
            ).fetchone()
            if row:
                title, section, text = row
                excerpts.append(f"[{pid}] {title} ({section})\n{text[:self.SNIPPET_CHARS]}")
        context = "\n\n".join(excerpts) if excerpts else "(no relevant excerpts found)"

        history = self.history.get(session_id, [])[-2 * self.HISTORY_TURNS:]
        user_turn = f"Course paper excerpts:\n\n{context}\n\nQuestion: {message}"
        return [{"role": "system", "content": system}, *history, {"role": "user", "content": user_turn}]

    def _generate(self, messages: list[dict], max_tokens: int | None = None) -> str:
        """
        calls the chat completions endpoint
        raises TimeoutError if the endpoint is unreachable, slow, or returns an error
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": max_tokens or self.max_tokens,
        }
        try:
            resp = requests.post(self._chat_url(), json=payload, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content")
        except requests.RequestException as exc:
            raise TimeoutError(f"LLM endpoint request failed: {exc}") from exc
        return (content or "").strip()
        
    
    def _build_query(self, session_id: str, message: str) -> str:
        """
        builds retrieval query according to self.query_mode:
          current: current message only
          window : current message plus last few user turns from same session
          rewrite: current message plus an LLM rewrite that resolves references to earlier turns
        """
        if self.query_mode == "current":
            return message
        if self.query_mode == "rewrite":
            return self._rewrite_query(session_id, message)
        return self._window_query(session_id, message)

    def _window_query(self, session_id: str, message: str) -> str:
        """
        concatenates the last QUERY_WINDOW - 1 user turns with the current message
        """
        history = self.sessions.get(session_id, [])
        n = self.QUERY_WINDOW - 1
        recent = history[-n:] if n > 0 else []
        return " ".join(recent + [message])

    def _rewrite_query(self, session_id: str, message: str) -> str:
        """
        asks the LLM for a standalone keyword query given recent dialogue
        first turn of a session is not rewritten (nothing to resolve)
        the original message is kept so a bad rewrite can only add terms, never drop them
        falls back to the current message alone if the rewrite is empty or too long,
        so rewrite on/off differs only in the rewrite itself
        """
        history = self.history.get(session_id, [])[-2 * self.REWRITE_TURNS:]
        if not history:
            return message

        lines = []
        for m in history:
            content = m["content"]
            if m["role"] == "assistant":
                content = content[:self.REWRITE_ANSWER_CHARS]
            lines.append(f"{m['role'].capitalize()}: {content}")
        prompt = (
            "Conversation so far:\n" + "\n".join(lines)
            + f"\n\nLatest question: {message}\n\nStandalone search query:"
        )
        rewrite = self._generate(
            [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=64,
        )

        ## keep first non-empty line, strip quotes and labels like "Query:"
        rewrite = next((l for l in rewrite.splitlines() if l.strip()), "").strip()
        rewrite = re.sub(r"^(standalone )?(search )?query\s*:\s*", "", rewrite, flags=re.I).strip(" \"'")
        if not rewrite or len(rewrite) > self.REWRITE_MAX_CHARS:
            return message
        return f"{message} {rewrite}"


    def _chat_url(self) -> str:
        """
        handles both url formats
        """
        base = self.endpoint.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return base + "/chat/completions"
