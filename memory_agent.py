"""
Multi-Memory Agent built with a LangGraph skeleton.

Graph topology:
    retrieve_memory → build_prompt → call_llm → save_memory → END

MemoryState carries:
    messages       — short-term conversation window
    user_id        — session identifier
    user_profile   — long-term profile facts
    episodes       — relevant episodic memories
    semantic_hits  — relevant semantic documents
    memory_budget  — remaining word-count budget for prompt
    built_prompt   — assembled system prompt (set by build_prompt node)
    pending_response — LLM reply (set by call_llm node)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph_skeleton import END, StateGraph
from memory_backends import (
    EpisodicMemory,
    LongTermProfile,
    SemanticMemory,
    ShortTermMemory,
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class MemoryState(TypedDict):
    messages: List[Dict[str, str]]
    user_id: str
    user_profile: Dict[str, Any]
    episodes: List[Dict[str, Any]]
    semantic_hits: List[str]
    memory_budget: int
    built_prompt: str
    pending_response: str


# ---------------------------------------------------------------------------
# Priority constants for token-budget trimming
# Priority 1 = highest (keep last); 4 = lowest (trim first)
# ---------------------------------------------------------------------------

PRIORITY_SEMANTIC  = 1  # domain knowledge — most reusable
PRIORITY_EPISODIC  = 2  # past experiences
PRIORITY_PROFILE   = 3  # user facts
PRIORITY_SHORT_TERM = 4  # raw recent turns (already windowed)

TOKEN_BUDGET = 1200       # word-count proxy for context window budget
SEMANTIC_BUDGET = 300     # max words from semantic hits
EPISODIC_BUDGET = 200     # max words from episodes
PROFILE_BUDGET  = 100     # max words from profile


# ---------------------------------------------------------------------------
# Fact-extraction patterns (simple regex, no LLM needed)
# ---------------------------------------------------------------------------

PROFILE_PATTERNS = [
    # name — requires 'là' keyword; stops at punctuation/end; blocks question words
    (r"tên\s+(?:\S+\s+){0,3}là\s+(\S+(?:\s+\S+){0,2})(?:\s*[.,!?]|$)", "name"),
    (r"(?:my name is|i(?:'m| am)\s+called?)\s+([A-Za-z][A-Za-z\s]{1,20}?)(?:\s*[.,!?]|$)", "name"),
    # allergy — declaration form only; stop at question words, 'chứ', comma, period, ?
    (r"(?:tôi\s+)?dị ứng\s+(\S+(?:\s+\S+)?)(?=\s*(?:chứ|nhưng|,|\.|!|\?|$))", "allergy"),
    (r"allerg(?:ic to|y to)\s+([a-z\s]+?)(?:\s*[,.]|$)", "allergy"),
    # language — explicit preference keyword required
    (r"(?:thích|prefer|dùng|use)\s+(python|javascript|typescript|go|rust|java|c\+\+)", "preferred_language"),
    # city
    (r"(?:sống ở|ở tại|live in|from)\s+(\S+(?:\s+\S+){0,2})(?:\s*[,.]|$)", "city"),
]

_NAME_BLOCKLIST = {"không", "gì", "sao", "đâu", "nào", "thế", "bao", "ai", "gi"}
_ALLERGY_BLOCKLIST = {"gì", "gi", "what", "nào", "của", "cua"}


CONFLICT_KEYS = {
    "allergy": "Allergy conflict",
    "name": "Name correction",
    "city": "Location update",
    "occupation": "Role change",
    "preferred_language": "Language preference update",
}


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def _call_openai(system_prompt: str, messages: List[Dict], model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API — used when OPENAI_API_KEY is set."""
    import openai  # noqa: PLC0415
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _mock_llm(system_prompt: str, messages: List[Dict]) -> str:
    """
    Deterministic context-aware mock LLM.

    Reads the injected memory sections from system_prompt and generates
    a response that clearly uses (or fails to use) the available context.
    This makes no-memory vs with-memory differences observable in benchmarks.
    """
    user_msg = messages[-1]["content"] if messages else ""
    user_lower = user_msg.lower()

    # Parse sections from the structured system prompt
    profile_section = _extract_section(system_prompt, "USER PROFILE")
    episodic_section = _extract_section(system_prompt, "PAST EPISODES")
    semantic_section = _extract_section(system_prompt, "KNOWLEDGE BASE")
    recent_section = _extract_section(system_prompt, "RECENT CONVERSATION")

    has_profile = bool(profile_section.strip())
    has_episodes = bool(episodic_section.strip())
    has_semantic = bool(semantic_section.strip())

    # --- Profile recall ---
    name_match = re.search(r"name[:\s]+([A-Za-zÀ-Ỹà-ỹ][A-Za-zÀ-Ỹà-ỹ\s]{1,25})", profile_section, re.I)
    allergy_match = re.search(r"allergy[:\s]+(.+?)(?:\n|$)", profile_section, re.I)
    lang_match = re.search(r"preferred_language[:\s]+(\w+)", profile_section, re.I)
    city_match = re.search(r"city[:\s]+(.+?)(?:\n|$)", profile_section, re.I)

    # ---- Route mock response by intent ----

    # Name recall
    if re.search(r"tên\s+tôi|my name|who am i|tôi là ai", user_lower):
        if name_match:
            name = name_match.group(1).strip()
            return f"Tên bạn là **{name}**. Tôi đã ghi nhớ từ trước đó."
        return "Xin lỗi, tôi không có thông tin về tên của bạn trong session này."

    # Allergy recall
    if re.search(r"dị ứng|allerg", user_lower):
        if allergy_match:
            allergy = allergy_match.group(1).strip()
            return f"Theo thông tin tôi lưu, bạn **dị ứng {allergy}**. Hãy tránh các thực phẩm liên quan."
        return "Tôi không tìm thấy thông tin dị ứng của bạn. Bạn có thể nhắc lại không?"

    # Language/tech preference recall
    if re.search(r"prefer|thích|recommend|nên dùng ngôn ngữ|which language", user_lower):
        if lang_match:
            lang = lang_match.group(1).strip()
            return f"Dựa trên sở thích của bạn, tôi recommend **{lang}** cho dự án này."
        return "Tôi chưa biết ngôn ngữ lập trình bạn ưa thích. Bạn có thể cho tôi biết không?"

    # Episodic recall — debug / previous lesson
    if re.search(r"trước|trước đó|remember|recall|last time|lần trước|debug|lỗi", user_lower):
        if has_episodes:
            # pull first episode summary
            ep_lines = [l for l in episodic_section.split("\n") if l.strip().startswith("-")]
            if ep_lines:
                return f"Lần trước chúng ta đã giải quyết: {ep_lines[0].lstrip('- ').strip()}. Bạn có muốn áp dụng lại phương pháp đó không?"
        return "Tôi không có ký ức về các cuộc hội thoại trước của chúng ta trong session này."

    # Semantic / knowledge base retrieval — broad pattern to catch policy/how-to queries
    if re.search(
        r"faq|policy|quy định|hướng dẫn|how to|làm thế nào|explain|giải thích"
        r"|chính sách|hoàn tiền|bảo hành|giao hàng|cài đặt|reset|mật khẩu",
        user_lower,
    ):
        if has_semantic:
            sem_lines = [l.strip().lstrip("- ") for l in semantic_section.split("\n") if l.strip()]
            if sem_lines:
                snippet = sem_lines[0][:200]
                return (
                    f"Theo knowledge base: **{snippet}**\n\n"
                    f"Bạn cần thêm thông tin về phần nào không?"
                )
        return "Tôi không tìm thấy tài liệu liên quan trong knowledge base của session này."

    # City recall
    if re.search(r"where.*live|ở đâu|thành phố|city", user_lower):
        if city_match:
            city = city_match.group(1).strip()
            return f"Bạn đang sống ở **{city}**. Tôi có thể hỗ trợ thông tin địa phương liên quan."
        return "Tôi không biết bạn đang sống ở đâu. Bạn có thể cho tôi biết không?"

    # Generic greeting with memory
    if re.search(r"^(xin chào|hello|hi|hey|chào)", user_lower):
        if name_match:
            return f"Xin chào **{name_match.group(1).strip()}**! Rất vui được gặp lại bạn. Tôi có thể giúp gì cho bạn hôm nay?"
        return "Xin chào! Tôi là AI assistant. Tôi có thể giúp gì cho bạn?"

    # Default: echo context summary
    context_parts = []
    if has_profile:
        context_parts.append("hồ sơ người dùng")
    if has_episodes:
        context_parts.append("ký ức về các cuộc hội thoại trước")
    if has_semantic:
        context_parts.append("knowledge base")

    if context_parts:
        return (
            f"Tôi đang xử lý yêu cầu của bạn dựa trên: {', '.join(context_parts)}. "
            f"Câu hỏi của bạn là: \"{user_msg}\". "
            f"Bạn có thể nói rõ hơn để tôi hỗ trợ chính xác hơn không?"
        )

    return (
        f"Tôi nhận được câu hỏi: \"{user_msg}\". "
        f"Hiện tại tôi chưa có đủ ngữ cảnh để trả lời chính xác. "
        f"Bạn có thể cung cấp thêm thông tin không?"
    )


def _extract_section(text: str, header: str) -> str:
    """Extract the content of a named section from the structured prompt."""
    pattern = rf"##\s+{re.escape(header)}\s*\n(.*?)(?=\n##|\Z)"
    m = re.search(pattern, text, re.S | re.I)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Memory Agent
# ---------------------------------------------------------------------------

class MultiMemoryAgent:
    """
    Agent with a 4-layer memory stack, orchestrated via a LangGraph skeleton.

    Backends
    --------
    short_term  : ShortTermMemory  (sliding window)
    long_term   : LongTermProfile  (JSON KV store / Redis-compatible)
    episodic    : EpisodicMemory   (timestamped JSON log)
    semantic    : SemanticMemory   (TF-IDF keyword search / Chroma-compatible)
    """

    def __init__(
        self,
        user_id: str = "user_001",
        data_dir: str = ".",
        use_memory: bool = True,
        llm_func=None,
    ) -> None:
        self.user_id = user_id
        self.use_memory = use_memory

        # Determine LLM backend
        if llm_func is not None:
            self._llm = llm_func
        elif os.environ.get("OPENAI_API_KEY"):
            self._llm = _call_openai
        else:
            self._llm = _mock_llm

        # Storage paths (per data_dir so parallel agents don't collide)
        pfx = os.path.join(data_dir, user_id)
        self.short_term = ShortTermMemory(max_messages=8)
        self.long_term  = LongTermProfile(f"{pfx}_profile.json")
        self.episodic   = EpisodicMemory(f"{pfx}_episodes.json")
        self.semantic   = SemanticMemory(f"{pfx}_semantic.json")

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(MemoryState)

        g.add_node("retrieve_memory", self._node_retrieve)
        g.add_node("build_prompt",    self._node_build_prompt)
        g.add_node("call_llm",        self._node_call_llm)
        g.add_node("save_memory",     self._node_save_memory)

        g.set_entry_point("retrieve_memory")
        g.add_edge("retrieve_memory", "build_prompt")
        g.add_edge("build_prompt",    "call_llm")
        g.add_edge("call_llm",        "save_memory")
        g.add_edge("save_memory",     END)

        return g.compile()

    # ------------------------------------------------------------------
    # Node: retrieve_memory
    # ------------------------------------------------------------------

    def _node_retrieve(self, state: dict) -> dict:
        """Pull from all 4 backends and merge into state."""
        if not self.use_memory:
            state["user_profile"]  = {}
            state["episodes"]      = []
            state["semantic_hits"] = []
            state["memory_budget"] = TOKEN_BUDGET
            return state

        user_id = state["user_id"]
        query   = state["messages"][-1]["content"] if state["messages"] else ""

        # Long-term profile
        profile = self.long_term.get_profile(user_id)

        # Episodic — search by query + get recent
        eps_search = self.episodic.search(query, user_id=user_id, limit=2)
        eps_recent = self.episodic.get_recent(user_id, limit=2)
        # Deduplicate by episode id
        seen_ids: set = set()
        episodes: List[dict] = []
        for ep in eps_search + eps_recent:
            if ep["id"] not in seen_ids:
                seen_ids.add(ep["id"])
                episodes.append(ep)

        # Semantic — search knowledge base
        sem_docs = self.semantic.query(query, top_k=3)
        semantic_hits = [d["content"] for d in sem_docs]

        state["user_profile"]  = profile
        state["episodes"]      = episodes
        state["semantic_hits"] = semantic_hits
        state["memory_budget"] = TOKEN_BUDGET
        return state

    # ------------------------------------------------------------------
    # Node: build_prompt
    # ------------------------------------------------------------------

    def _node_build_prompt(self, state: dict) -> dict:
        """Assemble context-rich system prompt with token-budget awareness."""
        budget = state["memory_budget"]
        sections: List[str] = []

        sections.append(
            "Bạn là một AI assistant hữu ích với khả năng ghi nhớ thông tin người dùng.\n"
            "Hãy sử dụng thông tin bên dưới để cá nhân hoá câu trả lời."
        )

        # --- Priority 3: User profile ---
        profile = {k: v for k, v in state["user_profile"].items()
                   if not k.startswith("_ts_")}
        if profile:
            profile_text = "\n".join(f"  {k}: {v}" for k, v in profile.items())
            profile_words = len(profile_text.split())
            if profile_words <= min(PROFILE_BUDGET, budget):
                sections.append(f"## USER PROFILE\n{profile_text}")
                budget -= profile_words

        # --- Priority 2: Episodic memories ---
        if state["episodes"]:
            ep_lines: List[str] = []
            for ep in state["episodes"]:
                line = f"  - [{ep['timestamp'][:10]}] {ep['summary']}"
                if ep.get("outcome"):
                    line += f" → {ep['outcome']}"
                ep_lines.append(line)
            ep_text = "\n".join(ep_lines)
            ep_words = len(ep_text.split())
            if ep_words <= min(EPISODIC_BUDGET, budget):
                sections.append(f"## PAST EPISODES\n{ep_text}")
                budget -= ep_words

        # --- Priority 1: Semantic / knowledge base ---
        if state["semantic_hits"]:
            sem_chunks: List[str] = []
            sem_budget = min(SEMANTIC_BUDGET, budget)
            used = 0
            for chunk in state["semantic_hits"]:
                words = len(chunk.split())
                if used + words > sem_budget:
                    break
                sem_chunks.append(f"  - {chunk}")
                used += words
            if sem_chunks:
                sections.append(f"## KNOWLEDGE BASE\n" + "\n".join(sem_chunks))
                budget -= used

        # --- Priority 4: Recent conversation (already windowed) ---
        recent = state["messages"][:-1]  # exclude current user turn
        if recent:
            conv_lines: List[str] = []
            conv_budget = budget
            used = 0
            for msg in reversed(recent):
                line = f"  {msg['role'].upper()}: {msg['content']}"
                words = len(line.split())
                if used + words > conv_budget:
                    break
                conv_lines.insert(0, line)
                used += words
            if conv_lines:
                sections.append(f"## RECENT CONVERSATION\n" + "\n".join(conv_lines))

        state["built_prompt"] = "\n\n".join(sections)
        state["memory_budget"] = budget
        return state

    # ------------------------------------------------------------------
    # Node: call_llm
    # ------------------------------------------------------------------

    def _node_call_llm(self, state: dict) -> dict:
        response = self._llm(state["built_prompt"], state["messages"])
        state["pending_response"] = response
        return state

    # ------------------------------------------------------------------
    # Node: save_memory
    # ------------------------------------------------------------------

    def _node_save_memory(self, state: dict) -> dict:
        """
        Extract facts from the conversation and persist them.

        Conflict handling (last-write-wins):
          If user corrects a previous fact (e.g., allergy), the new value
          overwrites the old one in LongTermProfile.update().
        """
        if not self.use_memory:
            return state

        user_id  = state["user_id"]
        user_msg = state["messages"][-1]["content"] if state["messages"] else ""
        response = state["pending_response"]

        # --- Extract profile facts from user message ---
        new_facts = _extract_profile_facts(user_msg)
        if new_facts:
            old_profile = state["user_profile"]
            conflicts = []
            for key, new_val in new_facts.items():
                old_val = old_profile.get(key)
                if old_val and old_val != new_val:
                    conflicts.append(
                        f"{CONFLICT_KEYS.get(key, key)}: '{old_val}' -> '{new_val}'"
                    )
            self.long_term.update(user_id, new_facts)
            if conflicts:
                msg = f"[CONFLICT RESOLVED] {'; '.join(conflicts)}"
                print(msg.encode("ascii", errors="replace").decode("ascii"))

        # --- Save episodic memory when a task is resolved ---
        if _is_task_completion(user_msg, response):
            summary = _summarize_exchange(user_msg, response)
            tags = _extract_tags(user_msg + " " + response)
            self.episodic.add_episode(
                user_id=user_id,
                summary=summary,
                outcome=response[:120],
                tags=tags,
            )

        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send a message and receive a memory-augmented response."""
        self.short_term.add("user", user_message)

        state: dict = {
            "messages":        self.short_term.get(),
            "user_id":         self.user_id,
            "user_profile":    {},
            "episodes":        [],
            "semantic_hits":   [],
            "memory_budget":   TOKEN_BUDGET,
            "built_prompt":    "",
            "pending_response": "",
        }

        final_state = self._graph.invoke(state)
        response = final_state["pending_response"]
        self.short_term.add("assistant", response)
        return response

    def load_knowledge(self, documents: List[str]) -> None:
        """Pre-load semantic memory with domain knowledge chunks."""
        for doc in documents:
            self.semantic.add(doc)

    def reset_session(self) -> None:
        """Clear short-term memory (long-term persists across sessions)."""
        self.short_term.clear()

    def debug_state(self) -> Dict[str, Any]:
        return {
            "user_id":        self.user_id,
            "short_term":     self.short_term.get(),
            "profile":        self.long_term.get_profile(self.user_id),
            "episodes_count": len(self.episodic.get_recent(self.user_id, limit=100)),
            "semantic_docs":  len(self.semantic._docs),
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_profile_facts(text: str) -> Dict[str, str]:
    """
    Extract key-value profile facts from a user message using regex patterns.
    Returns only clearly stated facts to avoid false positives.
    """
    facts: Dict[str, str] = {}
    text_lower = text.lower()

    for pattern, key in PROFILE_PATTERNS:
        m = re.search(pattern, text_lower, re.I)
        if m and m.lastindex and m.lastindex >= 1:
            value = m.group(1).strip().rstrip(".,;! ")
            if not value or len(value) <= 1:
                continue
            # Block common question words from being captured as profile values
            first_word = value.split()[0]
            if key == "name" and first_word in _NAME_BLOCKLIST:
                continue
            if key == "allergy" and first_word in _ALLERGY_BLOCKLIST:
                continue
            facts[key] = value

    # Occupation detection via keyword lookup (no capture group — separate logic)
    occupation_map = [
        ("lập trình viên", "software developer"),
        ("kỹ sư phần mềm", "software engineer"),
        ("kỹ sư", "software engineer"),
        ("data scientist", "data scientist"),
        ("designer", "designer"),
        ("developer", "software developer"),
    ]
    for kw, role in occupation_map:
        if kw in text_lower:
            facts["occupation"] = role
            break

    # Explicit language mention: "tôi thích python"
    lang_direct = re.search(r"\b(python|javascript|typescript|go|rust|java)\b", text_lower)
    if lang_direct and "thích" in text_lower:
        facts["preferred_language"] = lang_direct.group(1)

    return facts


def _is_task_completion(user_msg: str, response: str) -> bool:
    """Heuristic: save an episode when a concrete question is answered."""
    completion_signals = [
        r"cảm ơn|thank|thanks|got it|hiểu rồi|ok|okay|perfect|tuyệt",
        r"\?",  # user asked a question → answer is an episode worth saving
    ]
    for sig in completion_signals:
        if re.search(sig, user_msg, re.I):
            return True
    return len(response) > 80  # long answer = substantive episode


def _summarize_exchange(user_msg: str, response: str) -> str:
    """Create a one-line episode summary from the exchange."""
    user_short = user_msg[:80].rstrip()
    return f"User hỏi: \"{user_short}\" — Agent trả lời về chủ đề liên quan"


def _extract_tags(text: str) -> List[str]:
    """Extract topic tags from text for episodic search."""
    tag_map = {
        "debug": ["debug", "lỗi", "error", "fix"],
        "docker": ["docker", "container", "service"],
        "python": ["python"],
        "javascript": ["javascript", "js", "node"],
        "allergy": ["dị ứng", "allerg"],
        "cooking": ["nấu", "ăn", "thức ăn", "cook", "food", "recipe"],
        "travel": ["du lịch", "travel", "trip", "city"],
        "health": ["sức khỏe", "health", "bệnh", "thuốc"],
        "profile": ["tên", "name", "thành phố", "city"],
    }
    text_lower = text.lower()
    tags: List[str] = []
    for tag, keywords in tag_map.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return tags
