"""Reviewer 2: an adversarial read of a finished run before anyone cites it.

Muse Wonder gets the run's manifest (protocol, integrity events, results, the
methods paragraph) and is asked to find reasons a careful reviewer would not
trust the numbers. Its findings are advice attached to the run; they never change a
number — only the integrity checks and a person can do that.
"""

from __future__ import annotations

import json
from typing import Any

from kappa.agents.protocol_agent import output_text
from kappa.sam.client import ModelApi

WONDER_MODEL = "muse-wonder-1.0"

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["no_concerns", "minor_concerns", "major_concerns"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["note", "concern", "major"]},
                    "topic": {"type": "string", "enum": ["design", "definition", "tracking", "statistics", "reporting"]},
                    "claim": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "string"}},
                    "suggestion": {"type": "string"},
                },
                "required": ["severity", "topic", "claim", "refs", "suggestion"],
            },
        },
    },
    "required": ["verdict", "findings"],
}

INSTRUCTIONS = """\
You are Reviewer 2 for a behavioural neuroscience paper. You receive the complete record of one automated measurement run.
Find every reason a careful reviewer would not accept these numbers as reported: protocol definitions that depart from the paradigm's conventions, zone or threshold choices that change conclusions, integrity events that the numbers might still depend on, values that look physically implausible for the species, missing information a methods section needs, statements in the methods paragraph not supported by the record.
Be specific and cite the ids of the results or integrity events you are talking about in `refs`. Do not invent problems: if something is fine, leave it out. If there is nothing material, return verdict no_concerns with an empty findings list."""


def _brief(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest without bulky evidence lists, so the review fits comfortably."""
    slim = dict(manifest)
    slim["results"] = [
        {k: v for k, v in r.items() if k != "evidence"} | {"evidence_count": len(r.get("evidence", []))}
        for r in manifest.get("results", [])
    ]
    return slim


def review(api: ModelApi, manifest: dict[str, Any], effort: str = "high") -> dict[str, Any]:
    body = {
        "model": WONDER_MODEL,
        "instructions": INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(_brief(manifest), indent=1)}],
            }
        ],
        "reasoning": {"effort": effort},
        "text": {"format": {"type": "json_schema", "name": "review", "schema": FINDINGS_SCHEMA}},
        "max_output_tokens": 20000,
    }
    result = json.loads(output_text(api.create(body)))
    result["model"] = WONDER_MODEL
    return result
