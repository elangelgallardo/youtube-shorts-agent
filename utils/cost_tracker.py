"""
Cost tracking utility.
Accumulates usage per agent and calculates estimated USD cost.

Pricing (as of 2025):
  Gemini 2.5 Flash Lite  : $0.075/1M input tokens, $0.30/1M output tokens
  Gemini Flash Image     : $0.04/image (1080x1920)
  Veo 3                  : $0.35/second of generated video
  Google Cloud TTS Studio: $16.00/1M characters
"""

# ── Pricing table (USD) ──────────────────────────────────────────────────────
_PRICES = {
    "gemini_input_per_1m":   0.075,
    "gemini_output_per_1m":  0.300,
    "imagen_per_image":      0.040,
    "veo_per_second":        0.350,
    "tts_per_1m_chars":     16.000,
}


def record_gemini(job, agent: str, input_tokens: int, output_tokens: int) -> None:
    e = job.costs.setdefault(agent, {})
    e["gemini_input_tokens"]  = e.get("gemini_input_tokens",  0) + input_tokens
    e["gemini_output_tokens"] = e.get("gemini_output_tokens", 0) + output_tokens


def record_imagen(job, count: int = 1) -> None:
    e = job.costs.setdefault("media", {})
    e["imagen_images"] = e.get("imagen_images", 0) + count


def record_veo(job, seconds: float) -> None:
    e = job.costs.setdefault("media", {})
    e["veo_seconds"] = e.get("veo_seconds", 0.0) + seconds


def record_tts(job, characters: int) -> None:
    e = job.costs.setdefault("media", {})
    e["tts_characters"] = e.get("tts_characters", 0) + characters


def calculate_total(job) -> dict:
    """Return per-agent and total costs."""
    breakdown = {}
    total = 0.0

    for agent, usage in job.costs.items():
        cost = 0.0
        cost += usage.get("gemini_input_tokens",  0) / 1_000_000 * _PRICES["gemini_input_per_1m"]
        cost += usage.get("gemini_output_tokens", 0) / 1_000_000 * _PRICES["gemini_output_per_1m"]
        cost += usage.get("imagen_images",         0) * _PRICES["imagen_per_image"]
        cost += usage.get("veo_seconds",         0.0) * _PRICES["veo_per_second"]
        cost += usage.get("tts_characters",        0) / 1_000_000 * _PRICES["tts_per_1m_chars"]
        breakdown[agent] = round(cost, 5)
        total += cost

    breakdown["TOTAL"] = round(total, 5)
    return breakdown


def save_costs(job) -> None:
    """Persist actual costs to the video_costs table."""
    import sqlite3
    import os
    db_path = os.path.join(os.path.dirname(__file__), "..", "state.db")
    bd = calculate_total(job)
    total = bd.get("TOTAL", 0.0)
    usage = job.costs
    images = usage.get("media", {}).get("imagen_images", 0)
    veo_s = usage.get("media", {}).get("veo_seconds", 0.0)
    tts_c = usage.get("media", {}).get("tts_characters", 0)
    gemini_usd = sum(
        usage.get(a, {}).get("gemini_input_tokens", 0) / 1_000_000 * _PRICES["gemini_input_per_1m"]
        + usage.get(a, {}).get("gemini_output_tokens", 0) / 1_000_000 * _PRICES["gemini_output_per_1m"]
        for a in usage
    )
    conn = sqlite3.connect(os.path.normpath(db_path))
    conn.execute("""
        INSERT OR REPLACE INTO video_costs
        (job_id, created_at, title, images, has_veo, tts_chars, estimated,
         images_usd, veo_usd, tts_usd, gemini_usd, total_usd)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        job.job_id,
        job.created_at[:10],
        job.plan.title_concept,
        images,
        int(veo_s > 0),
        tts_c,
        0,  # actual, not estimated
        images * _PRICES["imagen_per_image"],
        veo_s * _PRICES["veo_per_second"],
        tts_c / 1_000_000 * _PRICES["tts_per_1m_chars"],
        gemini_usd,
        total,
    ))
    conn.commit()
    conn.close()


def format_cost_report(job) -> str:
    bd = calculate_total(job)
    total = bd.pop("TOTAL")
    lines = ["Cost breakdown:"]
    for agent, cost in bd.items():
        usage = job.costs.get(agent, {})
        details = []
        if usage.get("gemini_input_tokens"):
            details.append(f"{usage['gemini_input_tokens']:,} in / {usage.get('gemini_output_tokens',0):,} out tokens")
        if usage.get("imagen_images"):
            details.append(f"{usage['imagen_images']} images")
        if usage.get("veo_seconds"):
            details.append(f"{usage['veo_seconds']:.1f}s Veo clip")
        if usage.get("tts_characters"):
            details.append(f"{usage['tts_characters']:,} TTS chars")
        detail_str = f" ({', '.join(details)})" if details else ""
        lines.append(f"  {agent:<12} ${cost:.4f}{detail_str}")
    lines.append(f"  {'TOTAL':<12} ${total:.4f}")
    return "\n".join(lines)
