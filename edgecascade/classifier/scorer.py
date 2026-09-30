"""High-speed Semantic Complexity Scorer.

Provides sub-10ms classification using structural heuristics, lexical entropy,
and vector similarity against Tier 1, Tier 2, and Tier 3 semantic archetypes.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ComplexityScoreResult:
    """Detailed complexity scoring metrics."""
    score: float  # Normalized 0.0 to 1.0
    tier: str     # "tier_1", "tier_2", or "tier_3"
    latency_ms: float
    factors: Dict[str, float] = field(default_factory=dict)
    matched_heuristics: List[str] = field(default_factory=list)


# Semantic archetype anchors for rapid vector projection
TIER_1_ARCHETYPES = [
    "summarize this article in 3 sentences",
    "extract the json fields name, age, city",
    "classify sentiment as positive, neutral, or negative",
    "fix the syntax error or typo in this snippet",
    "translate the following paragraph into French",
    "convert markdown table to csv format",
    "write a simple regex to match a date",
    "generate boilerplate python dataclass with 3 fields",
    "rewrite this sentence to sound more professional",
    "parse these logs and find lines with 404 status",
]

TIER_2_ARCHETYPES = [
    "write a fastAPI endpoint with pydantic validation and error handling",
    "refactor this python class to use dependency injection",
    "create a react component with state hooks and modal dialog",
    "write unit tests with mocks for this database repository",
    "explain how garbage collection works in modern runtimes",
    "implement binary search tree with insertion and deletion in Go",
    "write a shell script to automate docker container health checks",
    "debug why this async event loop is blocking in nodejs",
]

TIER_3_ARCHETYPES = [
    "design a distributed consensus protocol fault-tolerant against Byzantine failures",
    "architect a multi-region low-latency event sourcing engine with Raft",
    "formally verify the deadlock-free guarantees of this concurrent lockless queue",
    "prove the time complexity bounds of this graph partitioning algorithm",
    "perform a comprehensive cryptographic security audit on this zero-knowledge circuit",
    "analyze trade-offs between CAP theorem and PACELC in geo-replicated databases",
    "derive the mathematical formulation for policy gradient optimization in RL",
    "architect an enterprise microservices migration strategy under zero-downtime SLA",
]


class ComplexityScorer:
    """Sub-10ms semantic query classifier for dynamic tiering."""

    def __init__(
        self,
        tier_1_max_threshold: float = 0.35,
        tier_2_max_threshold: float = 0.70,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.tier_1_threshold = tier_1_max_threshold
        self.tier_2_threshold = tier_2_max_threshold
        self.weights = weights or {
            "length": 0.15,
            "reasoning_markers": 0.35,
            "code_complexity": 0.20,
            "semantic_similarity": 0.30,
        }

        self._compile_regexes()
        self._build_vocabulary_and_anchors()

    def _compile_regexes(self) -> None:
        """Compile regex patterns for lightning-fast marker detection."""
        # Tier 1 indicators (simple instructions, extraction, formatting)
        self._t1_markers = re.compile(
            r"(?i)\b(?:summarize|summary|classify|sentiment|json|extract|format|translate|"
            r"rephrase|spellcheck|grammar|capitalize|lowercase|convert\s+to|csv|boilerplate|"
            r"regex|quick\s+question|what\s+is\s+the\s+capital|who\s+wrote)\b"
        )

        # Tier 3 indicators (advanced systems, formal logic, rigorous proofs)
        self._t3_markers = re.compile(
            r"(?i)\b(?:formal\s+verification|byzantine|consensus|distributed\s+systems?|"
            r"system\s+design|architect(?:ure)?|deadlock|lockless|memory\s+barrier|"
            r"zero-knowledge|cryptographic\s+proof|derive\s+the|mathematical\s+proof|"
            r"paxos|raft\s+algorithm|asymptotic\s+bound|formal\s+spec|soundness\s+and\s+completeness|"
            r"security\s+audit|adversarial\s+attack|kernel\s+patch|ebpf|compiler\s+optimization)\b"
        )

        # Code block detector
        self._code_block_pattern = re.compile(r"```[a-zA-Z0-9_\-]*\n[\s\S]*?```")

        # Multi-step reasoning triggers
        self._reasoning_steps = re.compile(
            r"(?i)\b(?:step\s*by\s*step|first\.\.\.second|evaluate\s+trade-?offs?|"
            r"compare\s+and\s+contrast|edge\s+cases|in-depth\s+analysis|pros\s+and\s+cons)\b"
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple, ultra-fast word tokenizer."""
        return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

    def _build_vocabulary_and_anchors(self) -> None:
        """Build term-frequency index over reference archetypes for sub-1ms projection."""
        all_texts = TIER_1_ARCHETYPES + TIER_2_ARCHETYPES + TIER_3_ARCHETYPES
        vocab_set = set()
        for doc in all_texts:
            for token in self._tokenize(doc):
                vocab_set.add(token)

        self._vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}
        self._vocab_size = len(self._vocab)

        # Pre-compute centroid vectors for each tier
        self._t1_centroid = self._compute_centroid(TIER_1_ARCHETYPES)
        self._t2_centroid = self._compute_centroid(TIER_2_ARCHETYPES)
        self._t3_centroid = self._compute_centroid(TIER_3_ARCHETYPES)

    def _vectorize(self, text: str) -> np.ndarray:
        """Compute normalized term-frequency vector in sub-millisecond time."""
        vec = np.zeros(self._vocab_size, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            idx = self._vocab.get(token)
            if idx is not None:
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def _compute_centroid(self, docs: List[str]) -> np.ndarray:
        """Compute normalized average centroid vector for document collection."""
        vectors = [self._vectorize(doc) for doc in docs]
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-6:
            centroid /= norm
        return centroid

    def score(self, text: str, message_count: int = 1) -> ComplexityScoreResult:
        """Classify query text and return complexity score result."""
        start_time = time.perf_counter()

        factors: Dict[str, float] = {}
        matched_heuristics: List[str] = []

        # 1. Length Factor (Logarithmic saturation around 2000 chars)
        char_len = len(text)
        length_factor = min(1.0, math.log1p(char_len) / math.log1p(2500))
        factors["length"] = length_factor

        # 2. Reasoning Markers
        t1_matches = len(self._t1_markers.findall(text))
        t3_matches = len(self._t3_markers.findall(text))
        reasoning_step_matches = len(self._reasoning_steps.findall(text))

        reasoning_score = 0.5  # neutral baseline
        if t1_matches > 0:
            reasoning_score -= min(0.35, t1_matches * 0.15)
            matched_heuristics.append(f"tier1_pattern_matches:{t1_matches}")
        if t3_matches > 0:
            reasoning_score += min(0.45, t3_matches * 0.20)
            matched_heuristics.append(f"tier3_pattern_matches:{t3_matches}")
        if reasoning_step_matches > 0:
            reasoning_score += min(0.25, reasoning_step_matches * 0.10)
            matched_heuristics.append(f"reasoning_steps:{reasoning_step_matches}")

        reasoning_score = max(0.0, min(1.0, reasoning_score))
        factors["reasoning_markers"] = reasoning_score

        # 3. Code Complexity Factor
        code_blocks = self._code_block_pattern.findall(text)
        code_block_count = len(code_blocks)
        total_code_chars = sum(len(c) for c in code_blocks)

        code_score = 0.0
        if code_block_count > 0:
            code_score = min(1.0, 0.3 + (total_code_chars / 1500.0) * 0.7)
            matched_heuristics.append(f"code_blocks:{code_block_count}")
        elif re.search(r"(?i)\b(?:def\s+|class\s+|function\s+|SELECT\s+.*FROM|async\s+def)\b", text):
            code_score = 0.35
            matched_heuristics.append("inline_code_keywords")
        factors["code_complexity"] = code_score

        # 4. Semantic Projection Similarity
        query_vec = self._vectorize(text)
        q_norm = np.linalg.norm(query_vec)

        if q_norm > 1e-6:
            sim_t1 = float(np.dot(query_vec, self._t1_centroid))
            sim_t2 = float(np.dot(query_vec, self._t2_centroid))
            sim_t3 = float(np.dot(query_vec, self._t3_centroid))

            # Convert similarities into a linear scale [0, 1]
            # Higher sim_t3 pushes towards 1.0, higher sim_t1 pushes towards 0.0
            sem_score = 0.5 + (sim_t3 * 0.5) - (sim_t1 * 0.4)
            sem_score = max(0.0, min(1.0, sem_score))
        else:
            sem_score = 0.5  # Neutral fallback when no vocab match

        factors["semantic_similarity"] = sem_score

        # Multi-turn conversation bonus (multi-turn queries generally require larger context/reasoning)
        multi_turn_bonus = min(0.15, max(0.0, (message_count - 1) * 0.03))
        if multi_turn_bonus > 0:
            matched_heuristics.append(f"multi_turn_turns:{message_count}")

        # Weighted aggregate
        w = self.weights
        base_score = (
            w.get("length", 0.15) * factors["length"]
            + w.get("reasoning_markers", 0.35) * factors["reasoning_markers"]
            + w.get("code_complexity", 0.20) * factors["code_complexity"]
            + w.get("semantic_similarity", 0.30) * factors["semantic_similarity"]
            + multi_turn_bonus
        )

        # High complexity architectural / formal proof override
        if t3_matches >= 2 or (t3_matches >= 1 and reasoning_step_matches >= 1):
            base_score = max(base_score, 0.75)
            matched_heuristics.append("high_complexity_override")
        elif t1_matches >= 1 and t3_matches == 0 and code_block_count == 0:
            # Strong pattern match for simple extraction, translation, or summary
            base_score = min(base_score, 0.30)
            matched_heuristics.append("tier1_pattern_override")

        total_score = max(0.0, min(1.0, base_score))

        # Tier assignment
        if total_score < self.tier_1_threshold:
            tier = "tier_1"
        elif total_score < self.tier_2_threshold:
            tier = "tier_2"
        else:
            tier = "tier_3"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ComplexityScoreResult(
            score=round(total_score, 4),
            tier=tier,
            latency_ms=round(elapsed_ms, 2),
            factors={k: round(v, 4) for k, v in factors.items()},
            matched_heuristics=matched_heuristics,
        )

    def score_messages(self, messages: List[Dict[str, Any]]) -> ComplexityScoreResult:
        """Extract text from OpenAI messages array and score."""
        combined = []
        user_or_asst_turns = 0
        for msg in messages:
            role = msg.get("role", "")
            if role != "system":
                user_or_asst_turns += 1
            content = msg.get("content")
            if isinstance(content, str):
                combined.append(content)
            elif isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        combined.append(p.get("text", ""))

        text = "\n".join(combined)
        return self.score(text, message_count=max(1, user_or_asst_turns))

