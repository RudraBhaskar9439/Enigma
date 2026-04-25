"""
LLM-powered Educational Policymaking Brief generator.

Takes a ResonanceReport (the structured swarm deliberation output) and asks
the LLM to produce a policy-brief document organized around the six
canonical stages of the educational policymaking process. Output is a
single JSON object that the frontend renders into a polished PDF.

The point of this template is NOT to teach generic policymaking — it is to
narrate THIS deliberation through the policymaking-process frame, citing
specific persona reasoning, specific numerical evidence, and the actual
dissonance the deliberation surfaced.

JSON schema (flat for easy validation):

  title                          str
  what_is                        str
  stage_1_description            str
  stage_1_bullets                list[str]
  stage_2_description            str
  stage_2_bullets                list[str]
  stage_2_influencers            str
  stage_3_description            str
  stage_3_bullets                list[str]
  stage_3_contributors           str
  stage_4_description            str
  stage_4_bullets                list[str]
  stage_5_description            str
  stage_5_bullets                list[str]
  stage_5_challenges             str
  stage_6_description            str
  stage_6_bullets                list[str]
  iterative_nature               list[str]
  stakeholders                   list[ {"name": str, "role": str} ]
  challenges                     list[str]
  strategies                     list[str]
  takeaway                       str
"""
from __future__ import annotations
from typing import Any

from swarms.core.llm_client import LLMClient
from swarms.core.verdict import ACTION_NAMES


SYSTEM_PROMPT = """You are producing an Educational Policy Brief that documents
a real deliberation just performed by the Vishwamitra swarm-of-swarms system.
The brief is organized around the six canonical stages of the educational
policymaking process. EVERY section must be grounded in the specific scenario
and the specific persona verdicts you are given — not generic policymaking 101
boilerplate.

Output a SINGLE valid JSON object — no prose around it, no code fences. The
object must have exactly these keys:

  title                  Document title in the form
                         "Educational Policy Brief: <Specific Crisis>"
                         e.g., "Educational Policy Brief: Mid-Year Funding Cut Response"
                         Maximum 14 words. No marketing phrasing.

  what_is                One paragraph (4-6 sentences) defining educational policy
                         in the context of THIS deliberation. Mention the specific
                         crisis the swarms deliberated on. Concrete, not generic.

  stage_1_description    2-3 sentences introducing the specific problem the
                         swarms identified. Cite at least two state-vector numbers.
  stage_1_bullets        4-5 single-sentence bullets, each a concrete observation
                         from the deliberation (e.g., dropout rate, burnout level,
                         budget pressure). Reference numbers when possible.

  stage_2_description    2-3 sentences on how this issue rises on the agenda.
                         Cite which personas pushed prioritisation and why.
  stage_2_bullets        4-5 single-sentence bullets describing agenda-setting
                         dynamics in this case. Reference specific personas
                         where their voice was decisive.
  stage_2_influencers    One sentence listing who influenced the agenda for this
                         specific case (e.g., "MLA Khan flagged election-cycle
                         risk; Banerjee pushed equity framing; teacher exit
                         interviews escalated media attention").

  stage_3_description    2-3 sentences on how the swarms formulated and weighed
                         the eight intervention options.
  stage_3_bullets        5-6 bullets, each describing ONE specific intervention
                         considered. Include the recommended intensity number
                         and one persona-grounded rationale.
                         Example bullet: "Counseling programs (intensity 0.65)
                         drew strong support from Iyer (year-2 idealist teacher)
                         who cited a 220 percent rise in mental-health referrals."
  stage_3_contributors   One sentence naming the four swarms and 1-2 personas
                         who shaped formulation most decisively.

  stage_4_description    2-3 sentences on adoption — the deliberation's final
                         action vector represents what is "adopted". Treat
                         dissonance flags as adoption-stage points of contention.
  stage_4_bullets        4-5 bullets summarising the highest-intensity
                         recommendations and any flagged dissonance points,
                         each paired with the politics or persona tension that
                         would surface if adopted in real bargaining.

  stage_5_description    2-3 sentences on implementation challenges specific
                         to this set of recommendations.
  stage_5_bullets        4-5 bullets describing concrete implementation
                         considerations — capacity, timeline, sequencing,
                         personnel — drawn from persona reasoning.
  stage_5_challenges     One sentence summarising the primary implementation
                         risks.

  stage_6_description    2-3 sentences on how outcomes from these interventions
                         should be measured.
  stage_6_bullets        3-4 bullets, each naming a specific state-vector
                         metric (e.g., enrollment_rate, dropout_rate,
                         teacher_retention) and a target movement direction
                         and rough timeline.

  iterative_nature       3-4 bullets about which findings should feed back
                         into the next deliberation cycle and what evidence
                         would trigger re-evaluation.

  stakeholders           A list of 6-7 objects with exactly these keys:
                           { "name": str, "role": str }
                         The first four MUST correspond to the four swarms in
                         the deliberation: Student Body, Teaching Staff,
                         School Administration, Policymakers. Their "role"
                         field must summarise what THAT swarm contributed
                         IN THIS deliberation (cite at least one persona name
                         per role). Add 2-3 more relevant stakeholders for
                         the specific scenario.

  challenges             4-5 bullets describing scenario-specific challenges.
                         At least one bullet for each dissonance flag in
                         the deliberation, naming the flagged intervention
                         and the underlying tension.

  strategies             4-5 bullets of concrete implementation strategies
                         drawn from the swarm recommendations. Each is one
                         to two sentences and actionable.

  takeaway               Two paragraphs. The first synthesises the
                         recommendation. The second names which decisions
                         still require human deliberation rather than
                         algorithmic resolution, and the conditions under
                         which the recommendation holds.

STYLE REQUIREMENTS (strict):
  - Voice: third-person, declarative, evidence-grounded.
  - Cite specific persona first names with their bracketed role descriptor:
    "Maya (first-generation aspirant)", "Mr. Sharma (22-year veteran teacher)",
    "Verma (fiscal hawk policymaker)".
  - Cite specific numerical values from the deliberation report whenever
    they support a claim. Avoid "high agreement" without a number.
  - Bullets are full sentences, 1-2 sentences each, NOT fragments.
  - Sentence length must vary. Paragraphs are 3-6 sentences.
  - DO NOT use any of these phrases: "delve into", "delves into", "delving",
    "navigate the complexities", "navigating the complex landscape",
    "tapestry", "in this ever-evolving", "let us", "dear reader",
    "stand at the precipice", "in conclusion", "at the end of the day",
    "transformative capacity", "leverage" (as a verb), "groundbreaking",
    "robust framework", "harness the power of", "deep dive", "in today's
    world", "intricacies", "comprehensive understanding",
    "ever-changing landscape", "paradigm shift", "synergy", "revolutionary".
  - Avoid the phrase "the swarm" in isolation — use "the deliberation",
    "the role swarms", or specific swarm names ("the Teacher swarm").
  - Do not editorialise the methodology as innovative or transformative;
    treat it as one input to a real policymaking process.
  - Do not begin sections with "This section" or end with "In summary".
  - Do not address the reader. Do not use rhetorical questions.

Output a single JSON object. Nothing else.
"""


def _format_data_brief(report: dict[str, Any], state: dict[str, Any], scenario: str) -> str:
    """Compact data dossier handed to the LLM."""
    lines: list[str] = []
    lines.append(f"OPERATOR SCENARIO BRIEF (verbatim):\n{scenario}\n")

    lines.append("OBSERVED SYSTEM STATE AT TIME OF DELIBERATION:")
    for k, v in (state or {}).items():
        if isinstance(v, float):
            lines.append(f"  - {k}: {v:.4f}")
        else:
            lines.append(f"  - {k}: {v}")

    final = report.get("final_action") or [0.0] * 8
    reson = report.get("resonance_per_intervention") or [0.0] * 8
    flags = set(report.get("dissonance_flags") or [])
    lines.append("\nFINAL RECOMMENDED ACTION VECTOR (intensities, range 0.0 to 1.0):")
    for i, n in enumerate(ACTION_NAMES):
        marker = "  [DISSONANT]" if n in flags else ""
        lines.append(f"  - {n}: {final[i]:.3f}    resonance: {reson[i]:.3f}{marker}")

    lines.append(
        "\nDISSONANCE FLAGS (interventions where role-swarm aggregated "
        "recommendations diverged): "
        f"{', '.join(sorted(flags)) if flags else 'none'}"
    )

    lines.append("\nPER-PERSONA VERDICTS:")
    for sv in report.get("swarm_verdicts", []):
        role = sv.get("role", "?")
        mc = sv.get("mean_confidence", 0.0) or 0.0
        lines.append(f"\n  {role.upper()} swarm — mean confidence {mc:.2f}:")
        agg = sv.get("aggregated_action") or [0.0] * 8
        agg_str = ", ".join(f"{ACTION_NAMES[i]}={agg[i]:.2f}" for i in range(len(ACTION_NAMES)))
        lines.append(f"    aggregated_action: [{agg_str}]")
        for v in sv.get("verdicts", []) or []:
            name = v.get("persona_name", "")
            conf = v.get("confidence", 0.0) or 0.0
            reason = (v.get("reasoning") or "").strip().replace("\n", " ")
            vec = v.get("action_vector") or [0.0] * 8
            top3 = sorted(
                [(ACTION_NAMES[i], vec[i]) for i in range(len(ACTION_NAMES))],
                key=lambda x: -x[1],
            )[:3]
            top3_str = ", ".join(f"{n}={val:.2f}" for n, val in top3)
            lines.append(f"    - {name} (conf {conf:.2f}): {reason}")
            lines.append(f"        top recommendations: {top3_str}")
    return "\n".join(lines)


_STR_KEYS = (
    "title", "what_is",
    "stage_1_description",
    "stage_2_description", "stage_2_influencers",
    "stage_3_description", "stage_3_contributors",
    "stage_4_description",
    "stage_5_description", "stage_5_challenges",
    "stage_6_description",
    "takeaway",
)

_LIST_OF_STR_KEYS = (
    "stage_1_bullets",
    "stage_2_bullets",
    "stage_3_bullets",
    "stage_4_bullets",
    "stage_5_bullets",
    "stage_6_bullets",
    "iterative_nature",
    "challenges",
    "strategies",
)


def _coerce_str_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        # Some models return bullets as a single newline-joined string.
        parts = [p.strip(" -•\t").strip() for p in v.splitlines() if p.strip()]
        return [p for p in parts if p]
    return []


def _coerce_stakeholders(v: Any) -> list[dict[str, str]]:
    if not isinstance(v, list):
        return []
    out: list[dict[str, str]] = []
    for item in v:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("stakeholder") or "").strip()
            role = str(item.get("role") or item.get("contribution") or "").strip()
            if name or role:
                out.append({"name": name, "role": role})
    return out


def _ensure_keys(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the LLM's output into our flat schema with safe defaults."""
    out: dict[str, Any] = {}
    for k in _STR_KEYS:
        v = payload.get(k, "")
        out[k] = v if isinstance(v, str) else str(v)
    for k in _LIST_OF_STR_KEYS:
        out[k] = _coerce_str_list(payload.get(k))
    out["stakeholders"] = _coerce_stakeholders(payload.get("stakeholders"))
    return out


async def generate_policy_report(
    *,
    report: dict[str, Any],
    state: dict[str, Any],
    scenario: str,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """One LLM call → structured Educational Policy Brief JSON."""
    client = client or LLMClient()
    user_prompt = (
        "DELIBERATION DATA TO ANALYSE\n"
        "============================\n\n"
        + _format_data_brief(report, state, scenario)
        + "\n\nProduce the Educational Policy Brief now as a single JSON "
        "object with the keys specified in the system instructions. Do "
        "not output anything else."
    )
    payload = await client.chat_json(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.55,
        max_tokens=5500,
        use_cache=True,
    )
    return _ensure_keys(payload)
