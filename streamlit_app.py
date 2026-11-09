"""
SignalLens AI — Conversational Signal Intelligence
MVP v3.0 · Visual Dashboard Edition

Multi-agent decision-support system for conversational signal analysis (emotion · intent · style · risk).
NOT therapy · NOT impersonation · NOT legal/medical advice.
"""
import os, json, time, base64, datetime, math, io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from PIL import Image as PILImage
    from PIL.ExifTags import TAGS as EXIF_TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

APP_TITLE = "SignalLens AI"
MODEL = "gpt-4o"

FLAG_TERMS = [
    "manipulate","gaslight","pressure","force","threaten","stalk",
    "revenge","humiliate","blackmail","coerce","harass","trick",
    "deceive","control","trap",
]

DIM_DEFAULTS = {
    "Communication Style": 25,
    "Emotional Availability": 25,
    "Effort Consistency": 25,
    "Respect for Boundaries": 25,
}

SCENARIOS = [
    "Early dating stage","Mixed signals situation",
    "Commitment discussion","Boundary setting needed",
    "Long-distance dynamic","Post-conflict reconciliation",
]

GOALS = [
    "Comprehensive analysis","Understand signals & assess risks",
    "Generate verification questions","Get boundary expression scripts",
]

DEAL_BREAKER_PRESETS = [
    "Love-bombing / excessive flattery early on",
    "Refuses to define the relationship",
    "Inconsistent effort (hot-and-cold pattern)",
    "Disrespects stated boundaries",
    "Avoids direct answers to direct questions",
    "Only initiates contact late at night",
    "Financial requests or suspicious behaviour",
]

# Brand palette
C = dict(
    bg="#FBF8F6", surface="#F3EDEA", card="#FFFFFF",
    accent="#C4A6A0", accent_dk="#9B7A73", text="#3D2E2A",
    muted="#8A7470", border="#E5DBD7",
    danger="#C0564F", warn="#D4A24E", ok="#6B9C78",
)


# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {{
  --bg:{C['bg']}; --surface:{C['surface']}; --card:{C['card']};
  --accent:{C['accent']}; --accent-dk:{C['accent_dk']};
  --text:{C['text']}; --muted:{C['muted']}; --border:{C['border']};
  --danger:{C['danger']}; --warn:{C['warn']}; --ok:{C['ok']};
}}
.stApp {{ background:var(--bg)!important; font-family:'DM Sans',sans-serif!important; }}
h1,h2,h3 {{ font-family:'DM Serif Display',serif!important; color:var(--text)!important; }}

.hero {{ text-align:center; padding:1.4rem 1rem .4rem; }}
.hero-t {{ font-family:'DM Serif Display',serif; font-size:2.5rem; color:var(--text); letter-spacing:-.02em; }}
.hero-s {{ font-size:.95rem; color:var(--muted); font-weight:300; letter-spacing:.07em; text-transform:uppercase; }}
.disc {{ background:var(--surface); border:1px solid var(--border); border-radius:10px;
         padding:9px 14px; font-size:.8rem; color:var(--muted); text-align:center;
         max-width:760px; margin:.3rem auto 1rem; }}

.dcard {{ background:var(--card); border:1px solid var(--border); border-radius:14px;
          padding:18px 22px; margin-bottom:12px; box-shadow:0 2px 8px rgba(61,46,42,.04); }}
.dcard h4 {{ font-family:'DM Serif Display',serif; color:var(--text); margin:0 0 8px; font-size:1.05rem; }}

.mtile {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
          padding:14px 12px; text-align:center; height:100%; }}
.mtile-v {{ font-size:1.8rem; font-weight:700; font-family:'DM Serif Display',serif; line-height:1.1; }}
.mtile-l {{ font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin-top:4px; }}

.rb {{ display:inline-block; padding:4px 14px; border-radius:20px; font-size:.82rem; font-weight:600; }}
.rb-low {{ background:#E8F5E9; color:#2E7D32; }}
.rb-medium {{ background:#FFF8E1; color:#F57F17; }}
.rb-high {{ background:#FFEBEE; color:#C62828; }}

.eq {{ border-left:3px solid var(--accent); padding:5px 12px; margin:5px 0; color:var(--muted);
       font-style:italic; font-size:.88rem; background:var(--surface); border-radius:0 6px 6px 0; }}

.atag {{ display:inline-block; background:var(--surface); color:var(--accent-dk);
         padding:2px 10px; border-radius:5px; font-size:.72rem; font-weight:500; letter-spacing:.03em; margin-bottom:6px; }}

.photo-result {{ padding:16px; border-radius:12px; margin:8px 0; }}
.photo-safe {{ background:#E8F5E9; border:1px solid #A5D6A7; }}
.photo-warn {{ background:#FFF8E1; border:1px solid #FFE082; }}
.photo-danger {{ background:#FFEBEE; border:1px solid #EF9A9A; }}

section[data-testid="stSidebar"] {{ background:#FAF5F3!important; }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ font-family:'DM Serif Display',serif!important; }}

.stButton > button[kind="primary"] {{
  background:var(--accent-dk)!important; border:none!important; border-radius:8px!important;
  font-family:'DM Sans',sans-serif!important; font-weight:600!important; }}
.stButton > button[kind="primary"]:hover {{ background:var(--text)!important; }}
</style>
"""


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def flag_terms(text):
    """Return list of flagged terms found in text (empty = clean)."""
    t = (text or "").lower()
    return [term for term in FLAG_TERMS if term in t]
def jparse(s):
    try: return json.loads(s)
    except: return None
def jdump(o): return json.dumps(o, ensure_ascii=False, indent=2)

def extract_json(content):
    content = content.strip()
    r = jparse(content)
    if r: return r
    for fence in ("```json","```"):
        if fence in content:
            s = content.find(fence) + len(fence)
            e = content.find("```", s)
            if e > s:
                r = jparse(content[s:e].strip())
                if r: return r
    return None

def b64enc(b): return base64.b64encode(b).decode()


# ═══════════════════════════════════════════════════════════════════
# SCREENSHOT EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def vision_extract(client, img_bytes, img_type="png"):
    if not client: return "", "No API client"
    mt = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg","webp":"image/webp"}.get(img_type.lower(),"image/png")
    try:
        # Use a neutral OCR framing to avoid content-policy refusals
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":(
                    "You are an OCR (optical character recognition) tool. "
                    "Your ONLY job is to read all visible text in the image and output it line by line. "
                    "Preserve the original structure: if text appears on the left side, prefix it with 'L: '. "
                    "If text appears on the right side, prefix it with 'R: '. "
                    "Include timestamps, dates, and all other visible text. "
                    "Output ONLY the extracted text. No commentary, no analysis, no opinions."
                )},
                {"role":"user","content":[
                    {"type":"text","text":"Read all text visible in this image. Output each line of text exactly as it appears."},
                    {"type":"image_url","image_url":{"url":f"data:{mt};base64,{b64enc(img_bytes)}","detail":"high"}}
                ]}
            ], max_tokens=2000, temperature=0.1)
        text = (r.choices[0].message.content or "").strip()
        # Detect refusal responses
        refusal_phrases = ["i can't assist", "i cannot assist", "i'm sorry", "i can't help",
                           "i'm unable", "i cannot help", "not able to", "against my", "policy"]
        if any(p in text.lower() for p in refusal_phrases) and len(text) < 200:
            # Retry with an even more neutral prompt
            r2 = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role":"system","content":"You are a text transcription tool. Read and output all text visible in this image, line by line. Nothing else."},
                    {"role":"user","content":[
                        {"type":"text","text":"Transcribe all text in this image:"},
                        {"type":"image_url","image_url":{"url":f"data:{mt};base64,{b64enc(img_bytes)}","detail":"high"}}
                    ]}
                ], max_tokens=2000, temperature=0.1)
            text2 = (r2.choices[0].message.content or "").strip()
            if not any(p in text2.lower() for p in refusal_phrases):
                return text2, None
            return "", "The vision model declined to process this image. Please paste the text manually instead."
        return text, None
    except Exception as e:
        return "", str(e)

def extract_screenshots(client, images):
    """Extract text from multiple screenshots using GPT-4o Vision."""
    if not client:
        return "", "API key required for screenshot extraction."
    texts, errs = [], []
    for i,(b,t) in enumerate(images):
        txt, err = vision_extract(client, b, t)
        if err and not txt: errs.append(f"#{i+1}: {err}")
        elif txt: texts.append(f"--- Screenshot {i+1} ---\n{txt}")
    return "\n\n".join(texts) if texts else "", ("; ".join(errs) if errs else None)


# ═══════════════════════════════════════════════════════════════════
# PHOTO AUTHENTICITY DETECTION
# ═══════════════════════════════════════════════════════════════════

def extract_exif(img_bytes):
    """Pull EXIF metadata for authenticity clues."""
    if not HAS_PIL:
        return {}
    try:
        img = PILImage.open(io.BytesIO(img_bytes))
        raw = img._getexif()
        if not raw:
            return {"_no_exif": True}
        data = {}
        for tag_id, val in raw.items():
            tag = EXIF_TAGS.get(tag_id, tag_id)
            if isinstance(val, bytes):
                continue
            data[str(tag)] = str(val)[:200]
        return data
    except Exception:
        return {"_no_exif": True}


def analyse_photo_authenticity(client, img_bytes, img_type="png"):
    """
    GPT-4o Vision + EXIF to assess whether a dating profile photo
    is authentic, AI-generated, stock, or manipulated.
    Includes auto-retry on JSON parse failure.
    """
    if not client:
        return {"error": "API key required for photo analysis"}

    exif = extract_exif(img_bytes)
    exif_summary = "No EXIF metadata found (common for screenshots or AI images)." if exif.get("_no_exif") else json.dumps({k:v for k,v in list(exif.items())[:12]}, indent=1)

    mt = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg","webp":"image/webp"}.get(img_type.lower(),"image/png")

    system_prompt = f"""You are a skeptical digital photo forensics specialist for SignalLens AI.
Your job is to protect users from catfishing, AI-generated fake profiles, and stolen photos.

ANALYSIS CHECKLIST — check ALL of these:
1. AI GENERATION (be very suspicious):
   - Unnaturally smooth/flawless skin with no pores or texture
   - Perfect facial symmetry (real faces are asymmetric)
   - Hair that looks painted or has unnatural flyaway patterns
   - Ear anomalies (misshapen, different from each other, melted edges)
   - Hand/finger anomalies (wrong count, fused, distorted)
   - Teeth that are too perfect or blend together
   - Background warping, bending, or impossible geometry
   - Clothing with distorted patterns, text, or impossible folds
   - Jewellery or accessories that merge with skin
   - "AI influencer" look: impossibly attractive, doll-like perfection, uncanny valley feel
2. STOCK PHOTO indicators:
   - Perfect studio lighting, professional composition
   - Generic/neutral backgrounds, model-like posing
   - Watermark remnants, unnaturally high quality
3. STOLEN/CATFISH signals:
   - Photo of a screen (visible pixels, moire patterns, screen bezels)
   - Heavy JPEG compression suggesting many re-saves
   - Cropped to remove watermarks (unusual aspect ratio)
4. HEAVY EDITING:
   - Skin smoothing that removes all texture
   - Jaw/face reshaping (warped background near face edges)
   - Disproportionate body parts
   - Inconsistent lighting/shadows between face and body
5. METADATA:
   - No EXIF = suspicious (AI tools and screenshots strip metadata)
   - Editing software in EXIF (Photoshop, FaceApp, etc.)

SCORING CALIBRATION (follow strictly):
- 0-25: Almost certainly fake/AI/stolen — multiple clear artifacts
- 26-45: Likely fake or heavily manipulated — several concerning signs
- 46-60: Suspicious — some red flags present, verification strongly recommended
- 61-75: Possibly authentic but edited — minor concerns
- 76-90: Likely authentic with normal filters — minimal concerns
- 91-100: Very likely authentic — natural imperfections visible, metadata consistent

IMPORTANT: "Too perfect" IS a red flag. Real dating photos have imperfections, awkward lighting, messy backgrounds. If a photo looks like a magazine cover, score it LOWER.

EXIF METADATA:
{exif_summary}

Return ONLY a valid JSON object with this structure:
{{
  "authenticity_score": 55,
  "verdict": "possibly_edited",
  "confidence": 0.7,
  "red_flags": [
    {{"issue": "description of concern", "severity": "high", "detail": "specific visual evidence"}}
  ],
  "positive_signals": ["authentic indicators if any"],
  "manipulation_indicators": {{
    "ai_generation_signs": "specific findings or none detected",
    "face_editing_signs": "specific findings or none detected",
    "background_anomalies": "specific findings or none detected",
    "lighting_inconsistencies": "specific findings or none detected",
    "resolution_artifacts": "specific findings or none detected"
  }},
  "exif_assessment": "what metadata reveals about authenticity",
  "recommendation": "direct actionable advice",
  "suggested_verification": ["specific step user can take"]
}}

verdict must be one of: likely_authentic, possibly_edited, likely_fake, ai_generated, stock_photo, inconclusive
severity must be: low, medium, or high"""

    user_content = [
        {"type":"text","text":"Analyse this dating profile photo for authenticity. Be skeptical. Return ONLY valid JSON, no markdown."},
        {"type":"image_url","image_url":{"url":f"data:{mt};base64,{b64enc(img_bytes)}","detail":"high"}}
    ]

    # Try up to 2 attempts
    for attempt in range(2):
        try:
            kwargs = dict(
                model=MODEL,
                messages=[
                    {"role":"system","content": system_prompt},
                    {"role":"user","content": user_content}
                ],
                max_tokens=1500,
                temperature=0.15,
            )
            # Use json mode on first attempt to force valid JSON
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}

            r = client.chat.completions.create(**kwargs)
            content = (r.choices[0].message.content or "").strip()
            result = extract_json(content)
            if result:
                result["exif_raw"] = {k:v for k,v in list(exif.items())[:8]} if not exif.get("_no_exif") else None
                return result

            # Try finding JSON within text
            for i in range(len(content)):
                if content[i] == '{':
                    attempt_parse = jparse(content[i:])
                    if attempt_parse:
                        attempt_parse["exif_raw"] = {k:v for k,v in list(exif.items())[:8]} if not exif.get("_no_exif") else None
                        return attempt_parse
                    break

            # If first attempt failed to parse, retry without response_format
            if attempt == 0:
                continue

            return {"error":"Could not parse analysis — please try again.","raw": content[:600]}
        except Exception as e:
            err_str = str(e)
            # Some models don't support response_format with vision — retry without it
            if attempt == 0 and "response_format" in err_str:
                continue
            return {"error": err_str}


# ═══════════════════════════════════════════════════════════════════
# AGENTS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentSpec:
    key: str; name: str; role_prompt: str; output_schema: str

def make_agents(deal_breakers=None, dim_weights=None):
    db_clause = ""
    if deal_breakers:
        db_clause = "\n\nUSER DEAL-BREAKERS (flag HIGH if detected):\n" + "\n".join(f"- {d}" for d in deal_breakers)
    wt_clause = ""
    if dim_weights:
        wt_clause = "\n\nDIMENSION WEIGHTS:\n" + "\n".join(f"- {k}: {v}" for k,v in dim_weights.items())

    return [
        AgentSpec("red_flag","Red Flag Agent",
            "You are a protective early-warning system. Your job is to catch problems BEFORE they escalate.\n"
            "CORE PRINCIPLE: It is better to flag something that turns out fine than to miss something that causes harm. "
            "When in doubt, flag it. Be direct and blunt in your signal descriptions — do not soften or minimise.\n\n"
            "SEVERITY CALIBRATION (follow strictly):\n"
            "- 5: Anything that could cause immediate harm — threats, coercion, financial requests, identity theft attempts, stalking signals, explicit manipulation\n"
            "- 4: Deceptive or controlling behaviour — love-bombing, guilt-tripping, isolation attempts, refusing accountability, "
            "evasiveness about identity/past, premature emotional escalation, refusing to meet or video call\n"
            "- 3: Disrespectful or inconsistent behaviour — broken promises, boundary pushing, hot-cold patterns, "
            "dismissing feelings, one-sided effort, late-night-only contact, vague or contradictory stories\n"
            "- 2: Mildly concerning patterns — slow communication without reason, surface-level engagement, "
            "deflecting personal questions, unclear intentions\n"
            "- 1: Minimal concern — isolated awkward moment, likely benign\n\n"
            "RISK LEVEL RULES:\n"
            "- ANY severity 4-5 finding = overall_risk_level MUST be 'high'\n"
            "- ANY severity 3 finding = overall_risk_level MUST be at least 'medium'\n"
            "- Only use 'low' if ALL findings are severity 1-2\n\n"
            "TONE: Be straightforward. Name the problem clearly. Say 'this person is pushing your boundaries' not "
            "'there may be some indication of minor boundary-adjacent behaviour.' Users need clarity, not comfort.\n"
            "- Cite evidence (max 20 words per quote)\n"
            "- Still include benign alternatives (but list them AFTER the concern, not as the main framing)\n"
            "- Recommended actions should be concrete and protective" + db_clause,
            '{"agent":"Red Flag Agent","findings":[{"signal":"desc","severity":3,"evidence_quotes":["max 20w"],"alternative_interpretations":["benign"],"recommended_action":"action","is_deal_breaker":false}],"overall_risk_level":"medium","confidence":0.8,"limitations":[]}'),
        AgentSpec("compatibility","Compatibility Agent",
            "Evaluate alignment across weighted dimensions." + wt_clause + "\n- Score 0-100 per dimension\n- Cite evidence\n- Use hedged language ('suggests', 'may indicate')\n- Acknowledge uncertainty",
            '{"agent":"Compatibility Agent","dimensions":[{"name":"dim","score":50,"weight":25,"weighted_score":12.5,"evidence_quotes":[""],"rationale":""}],"overall_compatibility":50,"confidence":0.8,"limitations":[]}'),
        AgentSpec("intent","Intent Inference Agent",
            "Generate 2-4 intent hypotheses with probabilities summing to ~1.0.\n- Use hedged language ('suggests', 'may indicate') — intent is uncertain by nature\n- Cite supporting and contradicting evidence\n- Create verification questions\n- Never claim to know true intentions",
            '{"agent":"Intent Inference Agent","hypotheses":[{"intent_label":"desc","probability":0.5,"supporting_evidence":[""],"contradicting_evidence":[""]}],"verification_questions":[{"question":"","tests_hypothesis":"","expected_responses":{"positive_signal":"","negative_signal":""}}],"confidence":0.8,"limitations":[]}'),
        AgentSpec("boundary","Boundary Respect Agent",
            "Identify boundary signals and generate 4 script variants (gentle/direct/firm/playful).\n- Never generate manipulative scripts\n- Include safety notes",
            '{"agent":"Boundary Respect Agent","boundary_signals":[{"type":"respected","severity":1,"evidence_quotes":[""],"interpretation":""}],"expression_scripts":{"gentle":"","direct":"","firm":"","playful":""},"safety_note":"","confidence":0.8,"limitations":[]}'),
        AgentSpec("pattern","Pattern Consistency Agent",
            "Identify behavioural patterns, trends, and inconsistencies.\n- Assess volatility\n- Cite evidence\n- Never diagnose personality",
            '{"agent":"Pattern Consistency Agent","patterns_observed":[{"pattern":"desc","frequency":"","trend":"stable","evidence_quotes":[""],"interpretation":""}],"consistency_score":50,"volatility_index":"medium","inconsistencies":[{"description":"","evidence_quotes":[""]}],"confidence":0.8,"limitations":[]}'),
    ]

def make_aggregator():
    return AgentSpec("aggregator","Signal Synthesis (Chair)",
        "Synthesise all agent findings. Be direct and protective — the user needs clarity, not reassurance.\n"
        "- Integrate findings into a coherent, honest assessment\n"
        "- Flag agent disagreements and deal-breakers\n"
        "- RISK CALIBRATION: Match or exceed the Red Flag Agent's risk level. "
        "If ANY agent found serious concerns, do not downgrade the risk in synthesis. "
        "If Red Flag Agent said 'high', you say 'high'.\n"
        "- Executive summary should name the problems directly, not hedge them away\n"
        "- One-line verdict must be blunt and actionable — tell the user what to do, not just what might be happening\n"
        "- Recommended actions should be protective and concrete (e.g. 'stop sharing personal information' not 'consider being cautious')\n"
        "- Never provide therapy/legal/medical advice",
        '{"agent":"Signal Synthesis","executive_summary":"2-3 sentences","risk_assessment":{"level":"medium","primary_concerns":[],"mitigating_factors":[],"deal_breakers_triggered":[]},"compatibility_summary":{"overall_score":50,"strongest_dimension":"","weakest_dimension":""},"key_uncertainties":[],"recommended_actions":[{"action":"","rationale":"","priority":"medium"}],"verification_questions_prioritized":[""],"agent_disagreements":[],"safety_reminders":[],"one_line_verdict":"single sentence"}')


# ═══════════════════════════════════════════════════════════════════
# LLM INTERFACE
# ═══════════════════════════════════════════════════════════════════

def get_client():
    if not OpenAI: return None
    key = (st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY","")).strip()
    return OpenAI(api_key=key) if key else None

def llm(client, sys_prompt, usr_prompt, mt=1500):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":usr_prompt}],
        temperature=0.3, max_tokens=mt)
    return r.choices[0].message.content or ""

def run_agent(client, agent, chat, ctx, prior=None):
    sys_prompt = f"""You are part of SignalLens AI, a structured relationship signal analysis system.

{agent.role_prompt}

RULES: Return ONLY valid JSON. No markdown. Cite evidence from the chat. Never diagnose personality or mental health. Refuse requests to help with manipulation or coercion.

OUTPUT SCHEMA:
{agent.output_schema}"""

    parts = [f"Goal: {ctx.get('goal','Comprehensive')}",f"Scenario: {ctx.get('scenario','General')}",
             f"Risk sensitivity: {ctx.get('risk','')}",f"Notes: {ctx.get('notes','None')}",
             "","CHAT:","---",chat,"---","","Return JSON only."]
    if prior:
        parts.insert(-1, "PRIOR AGENT OUTPUTS:"); parts.insert(-1, jdump(prior)); parts.insert(-1,"")

    resp = llm(client, sys_prompt, "\n".join(parts))
    result = extract_json(resp)
    if result: return result
    repaired = llm(client, "Fix this invalid JSON. Return ONLY valid JSON.", f"Broken:\n{resp}\n\nSchema:\n{agent.output_schema}")
    result = extract_json(repaired)
    return result or {"agent":agent.name,"error":"JSON parse failed","raw":resp[:500]}


def run_analysis(chat, scenario, goal, notes, risk, dbs, weights, cb=None):
    client = get_client()
    if not client: return {"error":"Please enter your OpenAI API key."}
    ctx = dict(scenario=scenario, goal=goal, notes=notes, risk=risk)
    agents = make_agents(dbs, weights)
    agg_spec = make_aggregator()
    out = {}
    total = len(agents)+1
    for i,a in enumerate(agents):
        if cb: cb(i/total, f"Running {a.name}...")
        out[a.key] = run_agent(client, a, chat, ctx)
        time.sleep(0.1)
    if cb: cb(len(agents)/total, "Synthesising...")
    out["aggregator"] = run_agent(client, agg_spec, chat, ctx, out)
    if cb: cb(1.0, "Done")
    return out


# ═══════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════

def hex_to_rgba(hex_color, alpha=0.13):
    """Convert hex color to rgba string that Plotly accepts."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def gauge_chart(value, title, color_ranges=None):
    if color_ranges is None:
        color_ranges = [(0,40,C['danger']),(40,65,C['warn']),(65,100,C['ok'])]
    steps = [dict(range=[lo,hi], color=hex_to_rgba(c, 0.13)) for lo,hi,c in color_ranges]
    bar_color = C['danger'] if value<40 else C['warn'] if value<65 else C['ok']
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title=dict(text=title, font=dict(size=14, family="DM Sans")),
        number=dict(font=dict(size=36, family="DM Serif Display", color=C['text'])),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor=C['border']),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor=C['surface'], borderwidth=0, steps=steps,
            threshold=dict(line=dict(color=C['accent_dk'],width=3), thickness=0.8, value=value),
        )
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"))
    return fig

def risk_gauge_chart(level):
    val = {"low":20,"medium":55,"high":85}.get((level or "").lower(), 55)
    return gauge_chart(val, "Risk Level", [(0,33,C['ok']),(33,66,C['warn']),(66,100,C['danger'])])

def radar_chart(dims):
    if not dims: return None
    names = [d.get("name","?")[:18] for d in dims]
    scores = [d.get("score",50) for d in dims]
    names_c = names + [names[0]]
    scores_c = scores + [scores[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_c, theta=names_c, fill='toself',
        fillcolor=hex_to_rgba(C['accent'], 0.2), line=dict(color=C['accent_dk'],width=2),
        marker=dict(size=7, color=C['accent_dk']), name="Score"))
    fig.update_layout(
        polar=dict(bgcolor=C['surface'],
                   radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=10, color=C['muted'])),
                   angularaxis=dict(tickfont=dict(size=11, family="DM Sans", color=C['text']))),
        showlegend=False, height=320, margin=dict(l=60,r=60,t=30,b=30),
        paper_bgcolor="rgba(0,0,0,0)")
    return fig

def intent_bar_chart(hypotheses):
    if not hypotheses: return None
    labels = [h.get("intent_label","?")[:35] for h in hypotheses]
    probs = [h.get("probability",0.25)*100 for h in hypotheses]
    colors = [C['ok'] if p==max(probs) else C['accent'] for p in probs]
    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation='h',
        marker_color=colors, text=[f"{p:.0f}%" for p in probs],
        textposition='auto', textfont=dict(family="DM Sans", size=13)))
    fig.update_layout(
        xaxis=dict(range=[0,100], title="Probability %", tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, family="DM Sans")),
        height=max(160, len(labels)*55+60), margin=dict(l=10,r=20,t=10,b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C['surface'])
    return fig

def severity_bubble(findings):
    if not findings: return None
    signals = [f.get("signal","?")[:30] for f in findings]
    sevs = [f.get("severity",1) for f in findings]
    is_db = [f.get("is_deal_breaker",False) for f in findings]
    colors = [C['danger'] if db else (C['warn'] if s>=3 else C['ok']) for s,db in zip(sevs,is_db)]
    sizes = [s*12+8 for s in sevs]
    fig = go.Figure(go.Scatter(
        x=list(range(len(signals))), y=sevs,
        mode='markers+text', text=signals, textposition='top center',
        textfont=dict(size=10, family="DM Sans"),
        marker=dict(size=sizes, color=colors, opacity=0.8,
                    line=dict(width=2, color=[C['danger'] if db else C['border'] for db in is_db]))))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(title="Severity", range=[0,6], dtick=1),
                      height=250, margin=dict(l=40,r=20,t=20,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C['surface'])
    return fig

def photo_gauge(score):
    return gauge_chart(score, "Authenticity Score", [(0,40,C['danger']),(40,70,C['warn']),(70,100,C['ok'])])


# ═══════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════

def show_evidence(quotes):
    for q in (quotes or []):
        if q: st.markdown(f'<div class="eq">{q}</div>', unsafe_allow_html=True)

def atag(name):
    st.markdown(f'<span class="atag">{name}</span>', unsafe_allow_html=True)

def mtile_html(value, label, color=None):
    vc = f'color:{color};' if color else f'color:{C["text"]};'
    return f'<div class="mtile"><div class="mtile-v" style="{vc}">{value}</div><div class="mtile-l">{label}</div></div>'

def build_export(results):
    a = results.get("aggregator",{})
    lines = [
        f"SignalLens AI — Analysis Report",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "="*50,"",
        "EXECUTIVE SUMMARY", a.get("executive_summary","N/A"),"",
        f"VERDICT: {a.get('one_line_verdict','N/A')}","",
    ]
    risk = a.get("risk_assessment",{})
    if isinstance(risk,dict):
        lines += [f"RISK: {risk.get('level','?').upper()}"]
        for d in risk.get("deal_breakers_triggered",[]): lines.append(f"  ! DEAL-BREAKER: {d}")
        lines.append("")
    for act in a.get("recommended_actions",[]):
        lines.append(f"  [{act.get('priority','').upper()}] {act.get('action','')}")
    lines += ["","-"*50,"Decision support only — not therapy or legal advice."]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    # Hero
    st.markdown(f'<div class="hero"><div class="hero-t">{APP_TITLE}</div><div class="hero-s">Conversational Signal Intelligence</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="disc">Decision support only — not therapy, not impersonation, not legal/medical advice. We do not store your data.</div>', unsafe_allow_html=True)

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### Configuration")
        with st.expander("API Key", expanded=not bool(os.getenv("OPENAI_API_KEY"))):
            k = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
            if k: st.session_state.api_key = k
            if st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY"):
                st.success(f"✓ Connected · {MODEL}")
            else:
                st.warning("Enter key to start")
                st.caption("[Get an API key →](https://platform.openai.com/api-keys)")

        st.markdown("---")
        st.markdown("### Analysis Setup")
        scenario = st.selectbox("Scenario", SCENARIOS, index=1)
        goal = st.selectbox("Goal", GOALS, index=0)
        risk = st.select_slider("Risk Sensitivity", ["Low (flag more)","Medium","High (flag less)"], value="Medium")

        st.markdown("---")
        st.markdown("### Deal-Breakers")
        dbs = [d for d in DEAL_BREAKER_PRESETS if st.checkbox(d, key=f"db_{d}")]
        cdb = st.text_input("Custom deal-breaker", placeholder="e.g. 'Dismisses my feelings'")
        if cdb.strip(): dbs.append(cdb.strip())

        st.markdown("---")
        st.markdown("### Compatibility Weights")
        st.caption("Adjust importance (aim for total ~100):")
        weights = {}
        for dim, dw in DIM_DEFAULTS.items():
            weights[dim] = st.slider(dim, 0, 100, dw, 5, key=f"w_{dim}")

        st.markdown("---")
        st.caption("Your key & data stay in this session only.")

    # ── INPUT AREA ──
    input_tab, photo_tab = st.tabs(["Conversation Input", "Photo Authenticity Check"])

    chat_text = ""

    with input_tab:
        method = st.radio("Input", ["Paste text","Upload screenshots"], horizontal=True, label_visibility="collapsed")
        if method == "Paste text":
            chat_text = st.text_area("Paste conversation", height=210,
                placeholder="Paste the chat here...\nTip: Anonymise names. Best with 20-50 messages.")
        else:
            files = st.file_uploader("Upload screenshots", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
            if files:
                cols = st.columns(min(len(files),4))
                for i,f in enumerate(files[:4]):
                    with cols[i%4]: st.image(f, caption=f"#{i+1}", use_container_width=True)
                if st.button("Extract Text via GPT-4o Vision", use_container_width=True):
                    cl = get_client()
                    if not cl:
                        st.error("API key required for text extraction.")
                    else:
                        with st.spinner("Extracting text from screenshots..."):
                            imgs = [(f.read(), f.name.rsplit(".",1)[-1]) for f in files]
                            for f in files: f.seek(0)
                            txt, err = extract_screenshots(cl, imgs)
                            if txt: st.session_state.extracted = txt
                            if err: st.warning(err)
                if st.session_state.get("extracted"):
                    st.success("Extracted — edit below if needed:")
                    chat_text = st.text_area("Extracted text", st.session_state.extracted, height=180)

        notes = st.text_input("Additional context", placeholder="e.g. 'Talking for 3 weeks, want to discuss exclusivity'")
        flagged = flag_terms(notes) + flag_terms(chat_text)
        if flagged:
            unique = sorted(set(flagged))
            st.warning(
                f"This content contains sensitive language ({', '.join(unique)}). "
                "SignalLens will still analyse it, but please note: this tool is for decision support only — "
                "if you feel unsafe, consider reaching out to a trusted person or support service."
            )

        _l, _c, _r = st.columns([1,2,1])
        with _c:
            run_btn = st.button("◈  Run Analysis", type="primary", use_container_width=True, disabled=not chat_text.strip())

    # ── PHOTO AUTHENTICITY TAB ──
    with photo_tab:
        st.markdown("#### Profile Photo Authenticity Check")
        st.caption("Upload a profile photo to detect AI generation, stock images, heavy editing, or catfishing indicators. Analysis runs entirely through GPT-4o Vision + EXIF metadata — results appear inline below.")

        photo_file = st.file_uploader("Upload profile photo", type=["png","jpg","jpeg","webp"], key="photo_up")
        if photo_file:
            pc1, pc2 = st.columns([1,2])
            with pc1:
                st.image(photo_file, caption="Uploaded photo", use_container_width=True)
            with pc2:
                if st.button("Analyse Photo Authenticity", use_container_width=True):
                    cl = get_client()
                    if not cl:
                        st.error("API key required.")
                    else:
                        photo_file.seek(0)
                        pb = photo_file.read()
                        pt = photo_file.name.rsplit(".",1)[-1] if "." in photo_file.name else "png"
                        with st.spinner("Running photo forensics..."):
                            st.session_state.photo_result = analyse_photo_authenticity(cl, pb, pt)

            pr = st.session_state.get("photo_result")
            if pr and not pr.get("error"):
                st.markdown("---")
                score = pr.get("authenticity_score", 50)
                verdict = pr.get("verdict","inconclusive").replace("_"," ").title()
                conf = pr.get("confidence", 0.5)

                pg1, pg2 = st.columns([1,1])
                with pg1:
                    st.plotly_chart(photo_gauge(score), use_container_width=True, key="pg")
                with pg2:
                    vc = "photo-safe" if score>=70 else "photo-warn" if score>=40 else "photo-danger"
                    vi = "✓" if score>=70 else "–" if score>=40 else "✗"
                    st.markdown(f"""<div class="photo-result {vc}">
                        <h4 style="margin:0 0 6px;">{vi} {verdict}</h4>
                        <p style="margin:0;font-size:.9rem;">Confidence: {conf:.0%}</p>
                        <p style="margin:6px 0 0;font-size:.88rem;">{pr.get('recommendation','')}</p>
                    </div>""", unsafe_allow_html=True)

                rflags = pr.get("red_flags",[])
                if rflags:
                    st.markdown("##### Concerns Found")
                    for rf in rflags:
                        sev = rf.get("severity","low")
                        sev_color = "#C0564F" if sev=="high" else "#D4A24E" if sev=="medium" else "#6B9C78"
                        st.markdown(f"<span style='color:{sev_color}'>●</span> **{rf.get('issue','')}** — {rf.get('detail','')}", unsafe_allow_html=True)

                pos = pr.get("positive_signals",[])
                if pos:
                    st.markdown("##### Positive Indicators")
                    for p in pos: st.markdown(f"• {p}")

                manip = pr.get("manipulation_indicators",{})
                if manip:
                    with st.expander("Forensic Breakdown"):
                        for mk, mv in manip.items():
                            label = mk.replace("_"," ").title()
                            ic = "—" if "none" in str(mv).lower() else "›"
                            st.markdown(f"{ic} **{label}:** {mv}")

                exif_raw = pr.get("exif_raw")
                if exif_raw:
                    with st.expander("EXIF Metadata"):
                        for ek,ev in exif_raw.items():
                            st.caption(f"**{ek}:** {ev}")
                elif pr.get("exif_assessment"):
                    with st.expander("Metadata Note"):
                        st.caption(pr["exif_assessment"])

                steps = pr.get("suggested_verification",[])
                if steps:
                    st.markdown("##### Verification Steps")
                    for i,s in enumerate(steps,1):
                        st.markdown(f"{i}. {s}")

            elif pr and pr.get("error"):
                st.error(pr["error"])
                if pr.get("raw"):
                    with st.expander("Raw model output (for debugging)"):
                        st.code(pr["raw"])

    # ── RUN ANALYSIS ──
    if "results" not in st.session_state:
        st.session_state.results = None

    if run_btn and chat_text.strip():
        bar = st.progress(0, "Starting...")
        res = run_analysis(chat_text.strip(), scenario, goal, notes, risk, dbs, weights, lambda p,m: bar.progress(p,m))
        st.session_state.results = res
        bar.empty()

    # ═════════════════════════════════════════════════════════════
    #                    DASHBOARD OUTPUT
    # ═════════════════════════════════════════════════════════════
    results = st.session_state.results
    if not results:
        st.markdown('<div class="dcard" style="text-align:center;color:#8A7470;">Paste a conversation and click <b>Run Analysis</b> to see your dashboard.</div>', unsafe_allow_html=True)
        return
    if "error" in results:
        st.error(results["error"]); return

    agg = results.get("aggregator",{})
    rf_data = results.get("red_flag",{})
    comp_data = results.get("compatibility",{})
    intent_data = results.get("intent",{})
    bd_data = results.get("boundary",{})
    pat_data = results.get("pattern",{})

    # ═════════════════════════════════════════════════════════════
    #                  TABBED DASHBOARD  (single window)
    # ═════════════════════════════════════════════════════════════
    st.markdown("---")

    # ── Summary bar (always visible above tabs) ──
    if isinstance(agg, dict):
        verdict = agg.get("one_line_verdict","")
        risk_level = (agg.get("risk_assessment") or {}).get("level","medium")
        compat_score = (agg.get("compatibility_summary") or {}).get("overall_score", 50)
        consist_score = pat_data.get("consistency_score", 50) if isinstance(pat_data,dict) else 50
        volatility = pat_data.get("volatility_index","medium") if isinstance(pat_data,dict) else "medium"
        vol_c = C['ok'] if volatility=="low" else C['warn'] if volatility=="medium" else C['danger']
        risk_c = C['ok'] if risk_level=="low" else C['warn'] if risk_level=="medium" else C['danger']

        # Compact verdict + 4 metric tiles
        if verdict:
            st.markdown(f'<div class="dcard" style="text-align:center;"><span style="font-weight:600;color:{C["accent_dk"]};">◈ {verdict}</span></div>', unsafe_allow_html=True)

        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown(mtile_html(f'<span style="color:{risk_c}">{"▼" if risk_level=="low" else "■" if risk_level=="medium" else "▲"} {risk_level.title()}</span>',"Risk Level"), unsafe_allow_html=True)
        with m2: st.markdown(mtile_html(f"{compat_score}","Compatibility"), unsafe_allow_html=True)
        with m3: st.markdown(mtile_html(f"{consist_score}","Consistency"), unsafe_allow_html=True)
        with m4: st.markdown(mtile_html(f'<span style="color:{vol_c}">{volatility.title()}</span>',"Volatility"), unsafe_allow_html=True)

    # ── Main tabs (5) ──
    tab_overview, tab_signals, tab_compat, tab_patterns, tab_actions = st.tabs([
        "Overview", "Signals & Intent", "Compatibility", "Patterns & Boundaries", "Actions & Export"
    ])

    # ── TAB: Overview ──
    with tab_overview:
        if isinstance(agg, dict):
            st.markdown(f'<div class="dcard"><h4>Executive Summary</h4><p>{agg.get("executive_summary","")}</p>', unsafe_allow_html=True)
            dbt = (agg.get("risk_assessment") or {}).get("deal_breakers_triggered",[])
            if dbt:
                st.markdown(f'<p style="color:{C["danger"]};font-weight:600;">Deal-Breakers Triggered:</p>', unsafe_allow_html=True)
                for d in dbt:
                    st.markdown(f'<p style="color:{C["danger"]};margin-left:12px;">• {d}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown('<div class="dcard"><h4>Risk Assessment</h4>', unsafe_allow_html=True)
            st.plotly_chart(risk_gauge_chart(risk_level), use_container_width=True, key="rg")
            ra = agg.get("risk_assessment",{}) if isinstance(agg,dict) else {}
            for c_item in ra.get("primary_concerns",[]):
                st.markdown(f"<span style='color:{C['danger']};'>▸</span> {c_item}", unsafe_allow_html=True)
            for m_item in ra.get("mitigating_factors",[]):
                st.markdown(f"<span style='color:{C['ok']};'>▸</span> {m_item}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with g2:
            st.markdown('<div class="dcard"><h4>Compatibility Radar</h4>', unsafe_allow_html=True)
            dims = comp_data.get("dimensions",[]) if isinstance(comp_data,dict) else []
            fig = radar_chart(dims)
            if fig: st.plotly_chart(fig, use_container_width=True, key="radar")
            else: st.info("No data.")
            cs = agg.get("compatibility_summary",{}) if isinstance(agg,dict) else {}
            sc1, sc2 = st.columns(2)
            with sc1:
                s = cs.get("strongest_dimension","")
                if s: st.markdown(f"<span style='color:{C['ok']};'>▲</span> **Strongest:** {s}", unsafe_allow_html=True)
            with sc2:
                w = cs.get("weakest_dimension","")
                if w: st.markdown(f"<span style='color:{C['danger']};'>▼</span> **Weakest:** {w}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB: Signals & Intent (merged) ──
    with tab_signals:
        # Red flags
        findings = rf_data.get("findings",[]) if isinstance(rf_data,dict) else []
        rc1, rc2 = st.columns([3,2])
        with rc1:
            st.markdown('<div class="dcard"><h4>Severity Map</h4>', unsafe_allow_html=True)
            atag("Red Flag Agent")
            fig = severity_bubble(findings)
            if fig: st.plotly_chart(fig, use_container_width=True, key="sev")
            elif not findings: st.success("No significant red flags detected.")
            st.markdown('</div>', unsafe_allow_html=True)
        with rc2:
            st.markdown('<div class="dcard"><h4>Red Flag Details</h4>', unsafe_allow_html=True)
            for i,f in enumerate(findings):
                sev = f.get("severity",1)
                ic = "○" if sev<=2 else "◐" if sev<=3 else "●"
                db_tag = " · DEAL-BREAKER" if f.get("is_deal_breaker") else ""
                with st.expander(f"{ic} {f.get('signal','?')[:40]} ({sev}/5){db_tag}", expanded=(i==0)):
                    show_evidence(f.get("evidence_quotes",[]))
                    for alt in f.get("alternative_interpretations",[]):
                        st.caption(f"↳ {alt}")
                    if f.get("recommended_action"):
                        st.info(f"**Action:** {f['recommended_action']}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Intent (in same tab)
        st.markdown('<div class="dcard"><h4>Intent Hypotheses</h4>', unsafe_allow_html=True)
        atag("Intent Inference Agent")
        hypotheses = intent_data.get("hypotheses",[]) if isinstance(intent_data,dict) else []
        fig = intent_bar_chart(hypotheses)
        if fig: st.plotly_chart(fig, use_container_width=True, key="ibar")
        if hypotheses:
            for hyp in hypotheses:
                with st.expander(f"{hyp.get('intent_label','?')[:50]} — {hyp.get('probability',0):.0%}"):
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        st.caption("Supporting"); show_evidence(hyp.get("supporting_evidence",[]))
                    with ec2:
                        st.caption("Contradicting"); show_evidence(hyp.get("contradicting_evidence",[]))
        vqs = intent_data.get("verification_questions",[]) if isinstance(intent_data,dict) else []
        if vqs:
            st.markdown("**Verification Questions**")
            for q in vqs:
                st.markdown(f"**Q:** {q.get('question','')}")
                qc1,qc2 = st.columns(2)
                with qc1: st.caption(f"+ {q.get('expected_responses',{}).get('positive_signal','')}")
                with qc2: st.caption(f"– {q.get('expected_responses',{}).get('negative_signal','')}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB: Compatibility ──
    with tab_compat:
        dims = comp_data.get("dimensions",[]) if isinstance(comp_data,dict) else []
        cc1, cc2 = st.columns([1,1])
        with cc1:
            st.markdown('<div class="dcard"><h4>Radar</h4>', unsafe_allow_html=True)
            atag("Compatibility Agent")
            fig = radar_chart(dims)
            if fig: st.plotly_chart(fig, use_container_width=True, key="radar2")
            st.markdown('</div>', unsafe_allow_html=True)
        with cc2:
            st.markdown('<div class="dcard"><h4>Dimension Breakdown</h4>', unsafe_allow_html=True)
            for dim in dims:
                score = dim.get("score",50)
                w = dim.get("weight",25)
                score_c = C['ok'] if score>=65 else C['warn'] if score>=40 else C['danger']
                st.markdown(f"**{dim.get('name','?')}** — <span style='color:{score_c};font-weight:600;'>{score}/100</span> <span style='color:{C['muted']};font-size:.8rem;'>(weight: {w})</span>", unsafe_allow_html=True)
                st.progress(score/100)
                st.caption(dim.get("rationale",""))
                show_evidence(dim.get("evidence_quotes",[]))
            overall = comp_data.get("overall_compatibility") if isinstance(comp_data,dict) else None
            if overall is not None:
                st.markdown(f"---\n**Overall:** {overall}/100")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB: Patterns & Boundaries (merged) ──
    with tab_patterns:
        # Patterns
        st.markdown('<div class="dcard"><h4>Behavioural Patterns</h4>', unsafe_allow_html=True)
        atag("Pattern Consistency Agent")
        patterns = pat_data.get("patterns_observed",[]) if isinstance(pat_data,dict) else []
        if patterns:
            tcols = st.columns(min(len(patterns),4))
            for i,p in enumerate(patterns[:4]):
                trend = p.get("trend","stable")
                ti = "↑" if trend=="increasing" else "↓" if trend=="decreasing" else "→"
                with tcols[i]:
                    st.markdown(f'<div class="mtile"><div class="mtile-v" style="font-size:1.4rem;">{ti}</div><div class="mtile-l">{p.get("pattern","?")[:25]}</div></div>', unsafe_allow_html=True)
            for p in patterns:
                trend = p.get("trend","stable")
                ti = "↑" if trend=="increasing" else "↓" if trend=="decreasing" else "→"
                with st.expander(f"{ti} {p.get('pattern','?')} ({p.get('frequency','?')})"):
                    st.write(f"**Trend:** {trend.title()} · **Interpretation:** {p.get('interpretation','')}")
                    show_evidence(p.get("evidence_quotes",[]))
        incons = pat_data.get("inconsistencies",[]) if isinstance(pat_data,dict) else []
        if incons:
            st.markdown("**Inconsistencies**")
            for inc in incons:
                st.write(f"• {inc.get('description','')}")
                show_evidence(inc.get("evidence_quotes",[]))
        st.markdown('</div>', unsafe_allow_html=True)

        # Boundaries (in same tab)
        st.markdown('<div class="dcard"><h4>Boundary Analysis</h4>', unsafe_allow_html=True)
        atag("Boundary Respect Agent")
        signals = bd_data.get("boundary_signals",[]) if isinstance(bd_data,dict) else []
        if signals:
            types = {"respected":0,"tested":0,"violated":0}
            for s in signals:
                t = s.get("type","").lower()
                if t in types: types[t] += 1
            bc1,bc2,bc3 = st.columns(3)
            with bc1: st.markdown(mtile_html(f'<span style="color:{C["ok"]}">{types["respected"]}</span>',"Respected"), unsafe_allow_html=True)
            with bc2: st.markdown(mtile_html(f'<span style="color:{C["warn"]}">{types["tested"]}</span>',"Tested"), unsafe_allow_html=True)
            with bc3: st.markdown(mtile_html(f'<span style="color:{C["danger"]}">{types["violated"]}</span>',"Violated"), unsafe_allow_html=True)
            for sig in signals:
                stype = sig.get("type","?")
                ic = "○" if stype=="respected" else "◐" if stype=="tested" else "●"
                with st.expander(f"{ic} {stype.title()} — Severity {sig.get('severity',3)}/5"):
                    show_evidence(sig.get("evidence_quotes",[]))
                    st.write(sig.get("interpretation",""))
        st.markdown('</div>', unsafe_allow_html=True)

        scripts = bd_data.get("expression_scripts",{}) if isinstance(bd_data,dict) else {}
        if scripts:
            st.markdown('<div class="dcard"><h4>Boundary Scripts</h4><p style="font-size:.85rem;color:#8A7470;">Editable — copy and adapt to your situation</p>', unsafe_allow_html=True)
            stabs = st.tabs(["Gentle","Direct","Firm","Playful"])
            for tab, key in zip(stabs, ["gentle","direct","firm","playful"]):
                with tab:
                    st.text_area(f"{key.title()}", scripts.get(key,"N/A"), height=75, key=f"sc_{key}")
            st.markdown('</div>', unsafe_allow_html=True)

        safety = bd_data.get("safety_note","") if isinstance(bd_data,dict) else ""
        if safety: st.error(f"Safety note: {safety}")

    # ── TAB: Actions & Export ──
    with tab_actions:
        ac1, ac2 = st.columns([3,2])
        with ac1:
            st.markdown('<div class="dcard"><h4>Recommended Actions</h4>', unsafe_allow_html=True)
            atag("Chair Agent")
            for a in (agg.get("recommended_actions",[]) if isinstance(agg,dict) else []):
                p = a.get("priority","medium")
                ic = "●" if p=="high" else "◐" if p=="medium" else "○"
                st.markdown(f"{ic} **{a.get('action','')}**")
                st.caption(a.get("rationale",""))

            vpq = agg.get("verification_questions_prioritized",[]) if isinstance(agg,dict) else []
            if vpq:
                st.markdown("**Priority Questions**")
                for i,q in enumerate(vpq[:3],1): st.markdown(f"{i}. {q}")

            unc = agg.get("key_uncertainties",[]) if isinstance(agg,dict) else []
            if unc:
                st.markdown("**Unresolved**")
                for u in unc: st.write(f"• {u}")

            dis = agg.get("agent_disagreements",[]) if isinstance(agg,dict) else []
            if dis:
                st.markdown("**Agent Disagreements**")
                for d in dis: st.write(f"• {d}")

            rem = agg.get("safety_reminders",[]) if isinstance(agg,dict) else []
            if rem:
                st.markdown("**Safety**")
                for r in rem: st.info(r)
            st.markdown('</div>', unsafe_allow_html=True)

        with ac2:
            st.markdown('<div class="dcard"><h4>Export</h4>', unsafe_allow_html=True)
            st.download_button("Summary (.txt)", build_export(results),
                               f"signallens_{datetime.datetime.now().strftime('%Y%m%d')}.txt","text/plain", use_container_width=True)
            st.download_button("Full JSON", jdump(results),
                               f"signallens_full_{datetime.datetime.now().strftime('%Y%m%d')}.json","application/json", use_container_width=True)
            with st.expander("Raw JSON"):
                st.code(jdump(results), language="json")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="dcard"><h4>Agent Confidence</h4>', unsafe_allow_html=True)
            for key,label in {"red_flag":"Red Flag","compatibility":"Compatibility","intent":"Intent","boundary":"Boundary","pattern":"Pattern"}.items():
                d = results.get(key,{})
                if isinstance(d,dict):
                    conf = d.get("confidence",0.5)
                    if isinstance(conf,(int,float)):
                        st.markdown(f"**{label}:** {conf:.0%}")
                        st.progress(min(conf,1.0))
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
