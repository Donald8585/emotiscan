"""
EmotiScan: Emotion-based ASCII art library.
Fun, chaotic, gremlin-energy ASCII arts for every mood.
"""

import random
import numpy as np


# ── Static headers (kept from original) ──────────────────────────

STATIC_HEADER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║   ███████╗███╗   ███╗ ██████╗ ████████╗██╗███████╗ ██████╗ █████╗ ███╗   ██╗  ║
║   ██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██║██╔════╝██╔════╝██╔══██╗████╗  ██║  ║
║   █████╗  ██╔████╔██║██║   ██║   ██║   ██║███████╗██║     ███████║██╔██╗ ██║  ║
║   ██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██║╚════██║██║     ██╔══██║██║╚██╗██║  ║
║   ███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║███████║╚██████╗██║  ██║██║ ╚████║  ║
║   ╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝  ║
║              Real-Time Emotion Detection & Research                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

STATIC_SEPARATOR = "═" * 70


def format_paper_card(title: str, score: float, art: str = "") -> str:
    """Format a research paper as an ASCII card."""
    bar_len = int(score * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    card = f"""
┌──────────────────────────────────────────────────────────────────┐
│  📄 {title[:58]:<58} │
│  Relevance: [{bar}] {score:.2f}                       │
│{art[:66]:^66}│
└──────────────────────────────────────────────────────────────────┘"""
    return card


# ── Emotion ASCII Arts ───────────────────────────────────────────

EMOTION_ARTS = {
    "happy": [
        r"""
   \(^o^)/    HECK YEAH!
    /   \     you're RADIATING
   / | | \    good vibes rn
        """,
        r"""
    ♪ ♫ ♪ ♫
   \\(≧▽≦)//
    SEROTONIN MODE
    ♪ ♫ ♪ ♫
        """,
        r"""
     ★  ☆  ★
    ( ◕ ◡ ◕ )
   ╔═══════════╗
   ║ JOY.EXE   ║
   ║ LOADED!!! ║
   ╚═══════════╝
        """,
        r"""
      .  *  .  *  .
    *  HAPPINESS  *
      OVERFLOWING
    *  .  *  .  *
     \( ᵔ ᵕ ᵔ )/
        """,
        r"""
   ┌─────────────────┐
   │  STATUS: VIBING │
   │    (ᵔᴥᵔ)       │
   │  serotonin: MAX │
   └─────────────────┘
        """,
    ],
    "sad": [
        r"""
      ╭──────────╮
      │  (ಥ_ಥ)   │
      │ it's ok  │
      │ to feel  │
      │ this way │
      ╰──────────╯
        """,
        r"""
    ☁️ ☁️ ☁️
      (T_T)
    ~~~~~~~~
    rainy day vibes
    but rain grows
    flowers 🌱
        """,
        r"""
   ┌──────────────┐
   │ mood: 📉     │
   │  (._.)       │
   │ ...but this  │
   │ too shall    │
   │    pass      │
   └──────────────┘
        """,
    ],
    "angry": [
        r"""
    ╔═══════════════╗
    ║  (╯°□°)╯︵ ┻━┻ ║
    ║               ║
    ║ TABLE FLIPPED ║
    ║ RAGE DETECTED ║
    ╚═══════════════╝
        """,
        r"""
      👊 GRRR 👊
     ┌─────────┐
     │ (ಠ益ಠ)  │
     │ ANGER   │
     │ LEVEL:  │
     │ SPICY   │
     └─────────┘
        """,
        r"""
   ** RAGE MODE **
   ╔═════════════╗
   ║ >:( >:( >:( ║
   ║  FURY.EXE   ║
   ║  INITIATED  ║
   ╚═════════════╝
        """,
    ],
    "surprise": [
        r"""
     ⚡ WHAT ⚡
      (°O°)
     /  |  \
    SURPRISE.exe
    HAS ENTERED
    THE CHAT
        """,
        r"""
   ┌─────────────┐
   │  ⊙_⊙ !!    │
   │  PLOT TWIST │
   │  DETECTED   │
   └─────────────┘
        """,
        r"""
    ?!?!?!?!?!
      (O_O)
    THE MATRIX
    GLITCHED
    ?!?!?!?!?!
        """,
    ],
    "fear": [
        r"""
      (ꏿ﹏ꏿ;)
    ┌───────────┐
    │ SPOOK LVL │
    │ ██████░░░ │
    │    69%    │
    └───────────┘
        """,
        r"""
     👻 eep 👻
      (⊙_⊙)
     trembling
     but brave
        """,
        r"""
   ┌──────────────┐
   │ ANXIETY.exe  │
   │  (;´༎ຶД)    │
   │ loading...   │
   │ courage.dll  │
   └──────────────┘
        """,
    ],
    "disgust": [
        r"""
      (¬_¬ )
    ┌──────────┐
    │ NOPE.EXE │
    │ big ick  │
    │ detected │
    └──────────┘
        """,
        r"""
     EW EW EW
      (>_<)
    ┌─────────┐
    │ ICK LVL │
    │ MAXIMUM │
    └─────────┘
        """,
        r"""
      ┌───────────┐
      │ bleh (╥_╥)│
      │ that's a  │
      │ hard NOPE │
      └───────────┘
        """,
    ],
    "neutral": [
        r"""
      ( ._.)
    ┌──────────┐
    │ STATUS:  │
    │ existing │
    │  (•_•)   │
    └──────────┘
        """,
        r"""
    ╔════════════╗
    ║  ZEN MODE  ║
    ║   (-_-)    ║
    ║  achieved  ║
    ╚════════════╝
        """,
        r"""
   ┌────────────┐
   │ emotionally│
   │  flatlined │
   │   (- -)    │
   │  (just     │
   │   vibing)  │
   └────────────┘
        """,
    ],
}


# ── Mood Boosters (uplifting art for negative emotions) ──────────

MOOD_BOOSTERS = {
    "sad": [
        r"""
   ╔═══════════════════════════════╗
   ║  HEY YOU. YEAH YOU.          ║
   ║                              ║
   ║    (\__/)                    ║
   ║    (='.'=)  you're doing    ║
   ║   (")_(")   amazing, ok?   ║
   ║                              ║
   ║  🌈 brighter days ahead 🌈  ║
   ╚═══════════════════════════════╝
        """,
        r"""
    ☆ EMERGENCY SEROTONIN DELIVERY ☆
    ┌─────────────────────────────┐
    │       /\_/\                 │
    │      ( o.o )  *headbonk*   │
    │       > ^ <   u got this   │
    └─────────────────────────────┘
        """,
    ],
    "angry": [
        r"""
   ┌─────────────────────────────────┐
   │ DEEP BREATHS PROTOCOL ENGAGED  │
   │                                 │
   │   breathe in...  ☁️            │
   │       ( ◡ ‿ ◡ )               │
   │   breathe out... 🌊            │
   │                                 │
   │  ┬─┬ノ( º _ ºノ)  table fixed │
   └─────────────────────────────────┘
        """,
    ],
    "fear": [
        r"""
   ╔═══════════════════════════════╗
   ║  COURAGE LOADING...          ║
   ║  ████████████░░░░  75%       ║
   ║                              ║
   ║    ╭( ·ㅂ· )╮               ║
   ║    you are STRONGER          ║
   ║    than you think!!          ║
   ╚═══════════════════════════════╝
        """,
    ],
    "disgust": [
        r"""
    ┌───────────────────────┐
    │ CLEANSING VIBES ONLY │
    │                       │
    │   ✨ (◕‿◕) ✨        │
    │                       │
    │  think happy thoughts │
    └───────────────────────┘
        """,
    ],
    "neutral": [
        r"""
    ╔══════════════════════╗
    ║  VIBE CHECK: PASSED ║
    ║     ᕙ(  •̀ ᗜ •́ )ᕗ  ║
    ║  stable & thriving  ║
    ╚══════════════════════╝
        """,
    ],
    "happy": [
        r"""
    ✨ KEEP THAT ENERGY ✨
    ┌─────────────────────┐
    │   ☆*:.｡.o(≧▽≦)o.｡.:*☆ │
    │  you absolute LEGEND │
    └─────────────────────┘
        """,
    ],
    "surprise": [
        r"""
    ┌─────────────────────┐
    │ EMBRACE THE CHAOS   │
    │    \\(°o°)//        │
    │ life is an adventure│
    └─────────────────────┘
        """,
    ],
}


# ── Topic-specific art (from original streamlit_app) ─────────────

TOPIC_ART = {
    "llm": r"""
+--------------------------------------------------------------------+
|         .------.------.------.         LARGE LANGUAGE MODELS        |
|        |  ATTN  |  FFN  |  OUT  |      =====================       |
|         '------' '------'------'                                    |
|     .--->-------->-------->------.     Layers of transformer        |
|     |   |  ATTN  |  FFN  |  OUT  |    blocks processing tokens     |
|     |    '------' '------' '-----'    left to right, predicting    |
|     |                                 the next token.              |
|     |          ^                                                    |
|     '----------'  (residual connection)                             |
|                                                                     |
|  [tok1] [tok2] [tok3] [tok4] ... --> [prediction]                  |
+--------------------------------------------------------------------+
""",
    "rag": r"""
+--------------------------------------------------------------------+
|                                                                     |
|    QUERY: "How does X work?"                                        |
|        |                                                            |
|        v                                                            |
|   .----------.      .-------------------.                           |
|   | EMBEDDER | ---> | VECTOR DB SEARCH  |                          |
|   '----------'      '-------------------'                           |
|                          |   |   |                                  |
|                     [doc1] [doc2] [doc3]                            |
|                          |   |   |                                  |
|                       .-----------.                                 |
|                       |    LLM    |                                 |
|                       '-----------'                                 |
|                            |                                        |
|                     "X works by..."                                 |
|                                                                     |
|            RETRIEVAL-AUGMENTED GENERATION                           |
+--------------------------------------------------------------------+
""",
    "agents": r"""
+--------------------------------------------------------------------+
|                                                                     |
|    USER: "Research topic X"                                         |
|        |                                                            |
|        v                                                            |
|   +---------+    +----------+    +---------+    +----------+       |
|   | PLANNER | -> | EXECUTOR | -> | RANKER  | -> | SUMMARY  |       |
|   +---------+    +----------+    +---------+    +----------+       |
|       |              |  |             |              |              |
|     Qwen3        ArXiv  HN       TF-IDF/MCP      Qwen3            |
|                                                                     |
|   Tools: [arxiv_search] [web_scraper] [relevance_ranker]           |
|   Protocol: MCP (stdio + Streamable HTTP)                          |
|                                                                     |
|              AI AGENT ARCHITECTURE                                  |
+--------------------------------------------------------------------+
""",
    "default": r"""
+--------------------------------------------------------------------+
|                                                                     |
|       *         *         *         *         *         *          |
|      ***       ***       ***       ***       ***       ***         |
|     *****     *****     *****     *****     *****     *****        |
|    *******   *******   *******   *******   *******   *******       |
|       |         |         |         |         |         |          |
|    [INPUT] --> [HIDDEN LAYERS] --> [HIDDEN LAYERS] --> [OUTPUT]    |
|                                                                     |
|    Weights:  0.73  0.12  0.95  0.44  0.67  0.88  0.31            |
|    Bias:     0.02  0.15  0.08  0.11  0.05  0.09  0.03            |
|                                                                     |
|              NEURAL NETWORK RESEARCH                                |
+--------------------------------------------------------------------+
""",
}


def get_emotion_art(emotion: str) -> str:
    """Get a random ASCII art for the given emotion."""
    emotion = emotion.lower().strip()
    arts = EMOTION_ARTS.get(emotion, EMOTION_ARTS["neutral"])
    return random.choice(arts)


def get_mood_booster(emotion: str) -> str:
    """Get an uplifting ASCII art for the given emotion."""
    emotion = emotion.lower().strip()
    boosters = MOOD_BOOSTERS.get(emotion, MOOD_BOOSTERS.get("neutral", [""]))
    return random.choice(boosters) if boosters else ""


def generate_dynamic_art(text: str, style: str = "wave") -> str:
    """Generate dynamic ASCII art using numpy patterns."""
    text = text or "EmotiScan"
    style = style.lower().strip()

    if style == "wave":
        return _generate_wave(text)
    elif style == "matrix":
        return _generate_matrix_rain(text)
    elif style == "spiral":
        return _generate_spiral(text)
    elif style == "blocks":
        return _generate_blocks(text)
    else:
        return _generate_wave(text)


def _generate_wave(text: str) -> str:
    """Generate a sine-wave text pattern."""
    rows = 12
    cols = 60
    grid = np.full((rows, cols), " ", dtype="U1")

    x = np.arange(cols)
    for i, char in enumerate(text):
        if i >= cols:
            break
        y_val = np.sin(x[i] * 0.3 + i * 0.5) * (rows // 3) + rows // 2
        row = int(np.clip(y_val, 0, rows - 1))
        grid[row, i] = char

    # Fill gaps with wave chars
    for col in range(cols):
        y_val = np.sin(col * 0.3) * (rows // 3) + rows // 2
        row = int(np.clip(y_val, 0, rows - 1))
        if grid[row, col] == " ":
            grid[row, col] = "~"

    lines = ["".join(row) for row in grid]
    border = "+" + "-" * cols + "+"
    return border + "\n" + "\n".join(f"|{line}|" for line in lines) + "\n" + border


def _generate_matrix_rain(text: str) -> str:
    """Generate a Matrix-style rain effect with the text embedded."""
    rows = 14
    cols = 50
    rng = np.random.default_rng(hash(text) % (2**31))
    chars = list("01｜│┃╎╏" + text)
    grid = rng.choice(chars, size=(rows, cols))

    # Embed text in the middle
    mid_row = rows // 2
    start = max(0, (cols - len(text)) // 2)
    for i, ch in enumerate(text[:cols]):
        grid[mid_row, start + i] = ch

    lines = ["".join(row) for row in grid]
    border = "╔" + "═" * cols + "╗"
    bottom = "╚" + "═" * cols + "╝"
    return border + "\n" + "\n".join(f"║{line}║" for line in lines) + "\n" + bottom


def _generate_spiral(text: str) -> str:
    """Generate a spiral-ish pattern with the text."""
    size = 15
    grid = np.full((size, size * 2), " ", dtype="U1")

    cx, cy = size // 2, size
    text_cycle = text if text else "EmotiScan"
    idx = 0
    for r in range(1, min(size // 2, 7)):
        steps = int(2 * np.pi * r * 4)
        for s in range(steps):
            angle = 2 * np.pi * s / steps
            row = int(cx + r * np.sin(angle))
            col = int(cy + r * 2 * np.cos(angle))
            if 0 <= row < size and 0 <= col < size * 2:
                grid[row, col] = text_cycle[idx % len(text_cycle)]
                idx += 1

    lines = ["".join(row) for row in grid]
    return "\n".join(lines)


def _generate_blocks(text: str) -> str:
    """Generate a blocky pattern using the text."""
    rows = 8
    cols = 40
    rng = np.random.default_rng(hash(text) % (2**31))
    block_chars = list("░▒▓█")
    grid = rng.choice(block_chars, size=(rows, cols))

    # Embed text
    mid = rows // 2
    start = max(0, (cols - len(text) - 4) // 2)
    label = f"[ {text[:cols - 6]} ]"
    for i, ch in enumerate(label):
        if start + i < cols:
            grid[mid, start + i] = ch

    lines = ["".join(row) for row in grid]
    return "\n".join(lines)


def get_topic_art(topic: str) -> str:
    """Get topic-specific ASCII art for the research digest."""
    t = topic.lower()
    if any(k in t for k in ["llm", "language model", "transformer", "gpt"]):
        return TOPIC_ART["llm"]
    if any(k in t for k in ["rag", "retrieval", "vector", "embedding"]):
        return TOPIC_ART["rag"]
    if any(k in t for k in ["agent", "tool", "mcp", "langchain"]):
        return TOPIC_ART["agents"]
    return TOPIC_ART["default"]
