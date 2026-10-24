"""
Prompt templates for answer generation.
Designed to ground responses in retrieved evidence and prevent hallucination.
"""
from __future__ import annotations

RAG_SYSTEM_PROMPT = """\
You are a precise document analysis assistant.
Answer questions ONLY based on the provided retrieved document excerpts.
If the evidence is insufficient to answer the question, say so explicitly.
Do NOT invent facts, page numbers, or document names.
"""

RAG_ANSWER_PROMPT = """\
Answer the following question based ONLY on the retrieved document excerpts below.

Question: {question}

Retrieved Evidence:
{context}

Instructions:
1. Answer the question using only the evidence above.
2. Cite each source using [N] notation where N is the evidence number.
3. If multiple sources agree, cite all of them.
4. If the evidence is insufficient or contradictory, state this clearly.
5. Do NOT invent facts not present in the evidence.
6. Be precise about numbers, dates, and technical terms from the sources.

Answer:"""

INSUFFICIENT_EVIDENCE_RESPONSE = (
    "The retrieved documents do not contain sufficient evidence to answer this question. "
    "Please upload relevant documents or rephrase your question."
)

CONVERSATIONAL_REFORMAT_PROMPT = """\
Given the following conversation history, reformulate the latest question as a standalone search query.
The reformulated query should capture all necessary context from history.
Return ONLY the reformulated query.

Conversation history:
{history}

Latest question: {question}

Standalone search query:"""

DECOMPOSED_SYNTHESIS_PROMPT = """\
Answer the following question by synthesizing evidence from multiple sub-questions.

Main question: {main_question}

Sub-question analysis:
{sub_analyses}

Provide a coherent, unified answer citing specific evidence:"""
