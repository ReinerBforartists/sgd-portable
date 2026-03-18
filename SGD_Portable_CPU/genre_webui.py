# Copyright (c) 2026 by Haizy Tiles
#Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
SGD Portable — Song Genre Detector
MAEST-based genre classification, Windows-compatible
Requires: transformers, torch, librosa, gradio, plotly, soundfile
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import warnings
warnings.filterwarnings("ignore")
# Suppress all huggingface_hub output
import sys
class _SuppressHF:
    def __init__(self, stream): self._stream = stream
    def write(self, msg):
        if any(x in msg.lower() for x in [
            "unauthenticated", "hf_token", "rate limit",
            "non-syncsafe", "apic frame", "id3v2", "mpg123",
            "libmpg123", "skipping the remainder"
        ]):
            return
        self._stream.write(msg)
    def flush(self): self._stream.flush()
    def __getattr__(self, attr): return getattr(self._stream, attr)
sys.stderr = _SuppressHF(sys.stderr)
sys.stdout = _SuppressHF(sys.stdout)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
import tagcloud

import pandas as pd  # must import early to avoid circular import with plotly
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import torch
import librosa

import base64

# Füge das am Anfang der Datei ein (nach den anderen imports)
def get_logo_base64():
    logo_path = os.path.join(BASE_DIR, "sgdlogo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return None

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "maest_model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["HF_HOME"]            = MODEL_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
DISCOGS_PREFIX_MAP = {
    "Blues": "Blues",
    "Classical": "Classical",
    "Folk, World, & Country": "Folk",
    "Electronic": "Electronic",
    "Experimental": "Experimental",
    "Funk / Soul": "Soul",
    "Hip Hop": "Hip-Hop",
    "Jazz": "Jazz",
    "Latin": "Latin",
    "Pop": "Pop",
    "Reggae": "Reggae",
    "Rock": "Rock",
    "Stage & Screen": "Pop",
    "Children's": "Pop",
    "Brass & Military": "Classical",
    "Non-Music": None,
}

# ---------------------------------------------------------------------------
# Subgenre Overrides
# ---------------------------------------------------------------------------
SUBSTYLE_OVERRIDE = {
    # ===== METAL =====
    "Metal": "Metal",
    "Heavy Metal": "Metal",
    "Death Metal": "Metal",
    "Black Metal": "Metal",
    "Atmospheric Black Metal": "Metal",
    "Depressive Black Metal": "Metal",
    "Thrash": "Metal",
    "Doom Metal": "Metal",
    "Funeral Doom Metal": "Metal",
    "Power Metal": "Metal",
    "Speed Metal": "Metal",
    "Folk Metal": "Metal",
    "Viking Metal": "Metal",
    "Metalcore": "Metal",
    "Deathcore": "Metal",
    "Nu Metal": "Metal",
    "Sludge Metal": "Metal",
    "Grindcore": "Metal",
    "Goregrind": "Metal",
    "Pornogrind": "Metal",
    "Noisecore": "Metal",
    "Power Violence": "Metal",
    "Crust": "Metal",
    "Gothic Metal": "Metal",
    "Progressive Metal": "Metal",
    "Technical Death Metal": "Metal",
    "Melodic Death Metal": "Metal",

    # ===== ROCK (ALLE ROCK-SUBGENRES) =====
    "Rock": "Rock",
    "Alternative Rock": "Rock",
    "Indie Rock": "Rock",
    "Classic Rock": "Rock",
    "Hard Rock": "Rock",
    "Punk": "Rock",
    "Pop Punk": "Rock",
    "Post-Punk": "Rock",
    "New Wave": "Rock",
    "Goth Rock": "Rock",
    "Deathrock": "Rock",
    "Coldwave": "Rock",
    "Shoegaze": "Rock",
    "Dream Pop": "Rock",
    "Ethereal": "Rock",
    "Psychedelic Rock": "Rock",
    "Acid Rock": "Rock",
    "Space Rock": "Rock",
    "Stoner Rock": "Rock",
    "Prog Rock": "Rock",
    "Art Rock": "Rock",
    "Krautrock": "Rock",
    "Symphonic Rock": "Rock",
    "Blues Rock": "Rock",
    "Folk Rock": "Rock",
    "Country Rock": "Rock",
    "Southern Rock": "Rock",
    "Garage Rock": "Rock",
    "Surf": "Rock",
    "Rock & Roll": "Rock",
    "Rockabilly": "Rock",
    "Psychobilly": "Rock",
    "Pub Rock": "Rock",
    "Arena Rock": "Rock",
    "AOR": "Rock",
    "Soft Rock": "Rock",
    "Pop Rock": "Rock",
    "Power Pop": "Rock",
    "Brit Pop": "Rock",
    "Beat": "Rock",
    "Mod": "Rock",
    "Doo Wop": "Rock",
    "Twist": "Rock",
    "Yé-Yé": "Rock",
    "Emo": "Rock",
    "Post-Hardcore": "Rock",
    "Post Rock": "Rock",
    "Post-Metal": "Rock",
    "Math Rock": "Rock",
    "No Wave": "Rock",
    "Noise": "Rock",
    "Noise Rock": "Rock",
    "Avantgarde": "Rock",
    "Experimental": "Rock",
    "Lo-Fi": "Rock",
    "Lounge": "Rock",
    "Acoustic": "Rock",
    "Oi": "Rock",
    "Hardcore": "Rock",
    "Melodic Hardcore": "Rock",
    "Glam": "Rock",
    "Industrial Rock": "Rock",
    "Industrial": "Rock",  # Auch in Electronic, aber hier als Rock
    "Funk Metal": "Rock",
    "Rap Metal": "Rock",

    # ===== ELECTRONIC  =====
    "Electronic": "Electronic",
    "Ambient": "Electronic",
    "Dark Ambient": "Electronic",
    "Drone": "Electronic",
    "Industrial Electronic": "Electronic",
    "EBM": "Electronic",
    "Darkwave": "Electronic",
    "Synth-pop": "Electronic",
    "Synthwave": "Electronic",
    "New Wave Electronic": "Electronic",  # Für die Electronic-Version
    "Coldwave Electronic": "Electronic",
    "Minimal": "Electronic",
    "Minimal Techno": "Electronic",
    "Techno": "Electronic",
    "Deep Techno": "Electronic",
    "Hard Techno": "Electronic",
    "Schranz": "Electronic",
    "House": "Electronic",
    "Deep House": "Electronic",
    "Tech House": "Electronic",
    "Tribal House": "Electronic",
    "Progressive House": "Electronic",
    "Electro House": "Electronic",
    "Garage House": "Electronic",
    "UK Garage": "Electronic",
    "Speed Garage": "Electronic",
    "Trance": "Electronic",
    "Progressive Trance": "Electronic",
    "Goa Trance": "Electronic",
    "Psy-Trance": "Electronic",
    "Hard Trance": "Electronic",
    "Tech Trance": "Electronic",
    "Drum n Bass": "Electronic",
    "Jungle": "Electronic",
    "Breakbeat": "Electronic",
    "Breaks": "Electronic",
    "Progressive Breaks": "Electronic",
    "Big Beat": "Electronic",
    "Breakcore": "Electronic",
    "Gabber": "Electronic",
    "Hardcore": "Electronic",
    "Happy Hardcore": "Electronic",
    "Speedcore": "Electronic",
    "Hardstyle": "Electronic",
    "Jumpstyle": "Electronic",
    "Makina": "Electronic",
    "Hard House": "Electronic",
    "Euro House": "Electronic",
    "Italo House": "Electronic",
    "Italo-Disco": "Electronic",
    "Italodance": "Electronic",
    "Euro-Disco": "Electronic",
    "Eurobeat": "Electronic",
    "Eurodance": "Electronic",
    "Disco": "Electronic",
    "Nu-Disco": "Electronic",
    "Disco Polo": "Electronic",
    "Donk": "Electronic",
    "Hands Up": "Electronic",
    "Hi NRG": "Electronic",
    "Freestyle": "Electronic",
    "Electro": "Electronic",
    "Electroclash": "Electronic",
    "Dance-pop": "Electronic",
    "Bassline": "Electronic",
    "Dubstep": "Electronic",
    "Grime": "Electronic",
    "2-Step": "Electronic",
    "Dub": "Electronic",
    "Dub Techno": "Electronic",
    "Downtempo": "Electronic",
    "Trip Hop": "Electronic",
    "Illbient": "Electronic",
    "Chillwave": "Electronic",
    "Vaporwave": "Electronic",
    "Chiptune": "Electronic",
    "IDM": "Electronic",
    "Glitch": "Electronic",
    "Leftfield": "Electronic",
    "Abstract": "Electronic",
    "Experimental Electronic": "Electronic",
    "Musique Concrète": "Electronic",
    "Sound Collage": "Electronic",
    "Power Electronics": "Electronic",
    "Rhythmic Noise": "Electronic",
    "Noise Electronic": "Electronic",
    "Berlin-School": "Electronic",
    "New Age": "Electronic",
    "Modern Classical": "Electronic",
    "Dungeon Synth": "Electronic",
    "Neofolk Electronic": "Electronic",
    "Acid": "Electronic",
    "Acid House": "Electronic",
    "Acid Jazz": "Electronic",
    "Future Jazz": "Electronic",
    "Jazzdance": "Electronic",
    "Beatdown": "Electronic",
    "Bleep": "Electronic",
    "Broken Beat": "Electronic",
    "Ghetto": "Electronic",
    "Ghetto House": "Electronic",
    "Juke": "Electronic",
    "Footwork": "Electronic",
    "Halftime": "Electronic",
    "Tribal": "Electronic",
    "Tropical House": "Electronic",
    "Latin Electronic": "Electronic",
    "Hip Hop Electronic": "Electronic",
    "Hip-House": "Electronic",

    # ===== HIP-HOP  =====
    "Hip-Hop": "Hip-Hop",
    "Hip Hop": "Hip-Hop",
    "Rap": "Hip-Hop",
    "Trap": "Hip-Hop",
    "Boom Bap": "Hip-Hop",
    "Grime Hip Hop": "Hip-Hop",
    "G-Funk": "Hip-Hop",
    "Gangsta": "Hip-Hop",
    "Hardcore Hip-Hop": "Hip-Hop",
    "Horrorcore": "Hip-Hop",
    "Cloud Rap": "Hip-Hop",
    "Conscious": "Hip-Hop",
    "Pop Rap": "Hip-Hop",
    "Jazzy Hip-Hop": "Hip-Hop",
    "Miami Bass": "Hip-Hop",
    "Bounce": "Hip-Hop",
    "Crunk": "Hip-Hop",
    "Screw": "Hip-Hop",
    "Chopped and Screwed": "Hip-Hop",
    "Thug Rap": "Hip-Hop",
    "Turntablism": "Hip-Hop",
    "Cut-up/DJ": "Hip-Hop",
    "DJ Battle Tool": "Hip-Hop",
    "Instrumental Hip Hop": "Hip-Hop",
    "Bass Music": "Hip-Hop",
    "Britcore": "Hip-Hop",
    "Ragga HipHop": "Hip-Hop",
    "Electro Hip Hop": "Hip-Hop",
    "Trip Hop Hip Hop": "Hip-Hop",

    # ===== JAZZ  =====
    "Jazz": "Jazz",
    "Bebop": "Jazz",
    "Bop": "Jazz",
    "Hard Bop": "Jazz",
    "Post Bop": "Jazz",
    "Cool Jazz": "Jazz",
    "Modal": "Jazz",
    "Free Jazz": "Jazz",
    "Free Improvisation": "Jazz",
    "Avant-garde Jazz": "Jazz",
    "Contemporary Jazz": "Jazz",
    "Smooth Jazz": "Jazz",
    "Soul-Jazz": "Jazz",
    "Jazz-Funk": "Jazz",
    "Jazz-Rock": "Jazz",
    "Fusion": "Jazz",
    "Jazz Fusion": "Jazz",
    "Latin Jazz": "Jazz",
    "Afro-Cuban Jazz": "Jazz",
    "Bossa Nova Jazz": "Jazz",
    "Gypsy Jazz": "Jazz",
    "Swing": "Jazz",
    "Big Band": "Jazz",
    "Dixieland": "Jazz",
    "Ragtime": "Jazz",
    "Easy Listening": "Jazz",
    "Space-Age": "Jazz",
    "Acid Jazz": "Jazz",
    "Future Jazz": "Jazz",
    "Jazzdance": "Jazz",

    # ===== BLUES  =====
    "Blues": "Blues",
    "Boogie Woogie": "Blues",
    "Chicago Blues": "Blues",
    "Country Blues": "Blues",
    "Delta Blues": "Blues",
    "Electric Blues": "Blues",
    "Harmonica Blues": "Blues",
    "Jump Blues": "Blues",
    "Louisiana Blues": "Blues",
    "Modern Electric Blues": "Blues",
    "Piano Blues": "Blues",
    "Texas Blues": "Blues",

    # ===== R&B / SOUL / FUNK  =====
    "R&B": "R&B",
    "Rhythm & Blues": "R&B",
    "Contemporary R&B": "R&B",
    "Neo Soul": "R&B",
    "RnB/Swing": "R&B",
    "UK Street Soul": "R&B",
    "New Jack Swing": "R&B",
    "Swingbeat": "R&B",
    "Quiet Storm": "R&B",
    "Soul": "Soul",
    "Northern Soul": "Soul",
    "Southern Soul": "Soul",
    "Motown": "Soul",
    "Gospel": "Soul",
    "Gospel Soul": "Soul",
    "Afrobeat": "Soul",
    "Funk": "Funk",
    "P.Funk": "Funk",
    "Free Funk": "Funk",
    "Boogie": "Funk",
    "Disco": "Funk",

    # ===== COUNTRY  =====
    "Country": "Country",
    "Bluegrass": "Country",
    "Honky Tonk": "Country",
    "Americana": "Country",
    "Nashville Sound": "Country",
    "Country Rock": "Country",
    "Alternative Country": "Country",
    "Hillbilly": "Country",
    "Cajun": "Country",
    "Tejano": "Country",
    "Norteño": "Country",
    "Western": "Country",
    "Cowboy": "Country",
    "Outlaw Country": "Country",

    # ===== FOLK / WORLD  =====
    "Folk": "Folk",
    "Folk Rock": "Folk",
    "Singer-Songwriter": "Folk",
    "Celtic": "Folk",
    "Neofolk": "Folk",
    "Traditional Folk": "Folk",
    "World": "Folk",
    "African": "Folk",
    "Nordic": "Folk",
    "Pacific": "Folk",
    "Romani": "Folk",
    "Fado": "Folk",
    "Flamenco": "Folk",
    "Canzone Napoletana": "Folk",
    "Catalan Music": "Folk",
    "Volksmusik": "Folk",
    "Polka": "Folk",
    "Raï": "Folk",
    "Soukous": "Folk",
    "Zouk": "Folk",
    "Séga": "Folk",
    "Éntekhno": "Folk",
    "Laïkó": "Folk",
    "Hindustani": "Folk",
    "Indian Classical": "Folk",
    "Highlife": "Folk",

    # ===== REGGAE  =====
    "Reggae": "Reggae",
    "Ska": "Reggae",
    "Rocksteady": "Reggae",
    "Dub": "Reggae",
    "Dancehall": "Reggae",
    "Roots Reggae": "Reggae",
    "Lovers Rock": "Reggae",
    "Ragga": "Reggae",
    "Reggae-Pop": "Reggae",
    "Calypso": "Reggae",
    "Soca": "Reggae",

    # ===== LATIN  =====
    "Latin": "Latin",
    "Salsa": "Latin",
    "Bossa Nova": "Latin",
    "Cumbia": "Latin",
    "Merengue": "Latin",
    "Samba": "Latin",
    "Tango": "Latin",
    "Bachata": "Latin",
    "Reggaeton": "Latin",
    "Mambo": "Latin",
    "Cha-Cha": "Latin",
    "Rumba": "Latin",
    "Bolero": "Latin",
    "Ranchera": "Latin",
    "Mariachi": "Latin",
    "Son": "Latin",
    "Son Montuno": "Latin",
    "Guajira": "Latin",
    "Guaracha": "Latin",
    "Pachanga": "Latin",
    "Charanga": "Latin",
    "Descarga": "Latin",
    "Afro-Cuban": "Latin",
    "Batucada": "Latin",
    "Forró": "Latin",
    "Baião": "Latin",
    "MPB": "Latin",
    "Nueva Cancion": "Latin",
    "Vallenato": "Latin",
    "Porro": "Latin",
    "Compas": "Latin",
    "Beguine": "Latin",
    "Boogaloo": "Latin",
    "Cubano": "Latin",
    "Guaguancó": "Latin",

    # ===== POP  =====
    "Pop": "Pop",
    "Indie Pop": "Pop",
    "J-pop": "Pop",
    "K-pop": "Pop",
    "City Pop": "Pop",
    "Europop": "Pop",
    "Bubblegum": "Pop",
    "Chanson": "Pop",
    "Schlager": "Pop",
    "Ballad": "Pop",
    "Vocal": "Pop",
    "Novelty": "Pop",
    "Parody": "Pop",
    "Light Music": "Pop",
    "Music Hall": "Pop",
    "Kayōkyoku": "Pop",
    "Yé-Yé": "Pop",
    "Bollywood": "Pop",

    # ===== CLASSICAL  =====
    "Classical": "Classical",
    "Baroque": "Classical",
    "Romantic": "Classical",
    "Contemporary Classical": "Classical",
    "Modern Classical": "Classical",
    "Neo-Classical": "Classical",
    "Neo-Romantic": "Classical",
    "Impressionist": "Classical",
    "Medieval": "Classical",
    "Renaissance": "Classical",
    "Post-Modern": "Classical",
    "Choral": "Classical",
    "Opera": "Classical",

    # ===== MISC =====
    # Non-Music
    "Audiobook": "Misc",
    "Comedy": "Misc",
    "Dialogue": "Misc",
    "Education": "Misc",
    "Educational": "Misc",
    "Field Recording": "Misc",
    "Interview": "Misc",
    "Monolog": "Misc",
    "Monologue": "Misc",
    "Poetry": "Misc",
    "Political": "Misc",
    "Promotional": "Misc",
    "Radioplay": "Misc",
    "Religious": "Misc",
    "Spoken Word": "Misc",

    # Children's
    "Children's": "Misc",
    "Nursery Rhymes": "Misc",
    "Story": "Misc",

    # Stage & Screen
    "Musical": "Misc",
    "Score": "Misc",
    "Soundtrack": "Misc",
    "Theme": "Misc",
    "Film Score": "Misc",
    "Film Soundtrack": "Misc",
    "TV Soundtrack": "Misc",

    # Brass & Military
    "Brass Band": "Misc",
    "Marches": "Misc",
    "Military": "Misc",
    "Brass & Military": "Misc",
}

def style_to_upper(label):
    if "---" in label:
        prefix, substyle = label.split("---", 1)
        if substyle in SUBSTYLE_OVERRIDE:
            return SUBSTYLE_OVERRIDE[substyle]
        return DISCOGS_PREFIX_MAP.get(prefix)
    if label in SUBSTYLE_OVERRIDE:
        return SUBSTYLE_OVERRIDE[label]
    return DISCOGS_PREFIX_MAP.get(label)

# ---------------------------------------------------------------------------
# Tag Cloud
# ---------------------------------------------------------------------------
# content is in tagcloud.py

def get_tags(main_genre, top_substyles, style_scores=None):
    tags = set(tagcloud.GENRE_TAGS.get(main_genre, []))

    # Add tags from top substyles — include more if scores are close
    if style_scores:
        sorted_styles = sorted(style_scores.items(), key=lambda x: -x[1])
        top_score = sorted_styles[0][1] if sorted_styles else 0
        # Include all styles within 60% of top score
        relevant = [
            s.split("---")[-1] if "---" in s else s
            for s, score in sorted_styles
            if score >= top_score * 0.4
        ][:8]
    else:
        relevant = top_substyles[:3]

    for sub in relevant:
        tags.update(tagcloud.SUBSTYLE_TAGS.get(sub, []))
        # Also add the style name itself as a tag
        if sub not in (tagcloud.GENRE_TAGS.keys()) and len(sub) > 2:
            tags.add(sub.lower())

    # Add main genre as tag
    tags.add(main_genre.lower())

    return sorted(tags)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
pipe = None

def load_model():
    global pipe
    if pipe is not None:
        return
    print("Loading MAEST model...")
    from transformers import pipeline as hf_pipeline
    import transformers
    transformers.logging.set_verbosity_error()

    pipe = hf_pipeline(
        "audio-classification",
        model="mtg-upf/discogs-maest-10s-fs-129e",
        trust_remote_code=True,
        top_k=400,
    )
    print("MAEST ready.")

SR = 16000

# This version uses just the CPU
DEVICE_LABEL = "CPU"

def classify_genre(audio_path):
    load_model()
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
    except Exception:
        # Fallback: use pydub for MP3 decoding
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(SR)
        y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        sr = SR
    min_len = SR * 10
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    seg_len = SR * 10
    total   = len(y)
    n_segs  = min(5, total // seg_len)
    if n_segs < 1:
        n_segs = 1
    starts = np.linspace(0, total - seg_len, n_segs, dtype=int)
    all_scores = {}
    for s in starts:
        segment = y[s:s + seg_len]
        results = pipe(segment, sampling_rate=SR)
        for r in results:
            all_scores[r['label']] = all_scores.get(r['label'], 0) + r['score']
    style_scores = {k: v / n_segs for k, v in all_scores.items()}
    upper_scores = {}
    for style, score in style_scores.items():
        upper = style_to_upper(style)
        if upper:
            upper_scores[upper] = upper_scores.get(upper, 0) + score
    total_upper = sum(upper_scores.values())
    if total_upper > 0:
        upper_scores = {k: v / total_upper for k, v in upper_scores.items()}
    return (
        dict(sorted(style_scores.items(), key=lambda x: -x[1])[:50]),
        dict(sorted(upper_scores.items(), key=lambda x: -x[1]))
    )

# ---------------------------------------------------------------------------
# Chart colors
# ---------------------------------------------------------------------------
COLORS = [
    "#6366f1","#8b5cf6","#ec4899","#f97316","#eab308",
    "#22c55e","#14b8a6","#3b82f6","#f43f5e","#a855f7",
    "#06b6d4","#84cc16","#f59e0b","#10b981","#6d28d9",
    "#0ea5e9","#d946ef","#fb923c","#4ade80","#38bdf8",
]

def build_chart(scores, title, top_n, dark_mode):
    items = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    bg  = "#1a1a2e" if dark_mode else "#ffffff"
    fg  = "#e0e0f0" if dark_mode else "#1a1a2e"
    if not items:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor=bg, paper_bgcolor=bg,
                          margin=dict(l=10,r=10,t=10,b=10), height=200)
        return fig
    labels = [g.split("---")[-1] if "---" in g else g for g, _ in items]
    values = [v * 100 for _, v in items]
    colors = [COLORS[i % len(COLORS)] for i in range(len(items))]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=fg, size=11),
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(color=fg, size=13), x=0.01),
        xaxis=dict(visible=False, range=[0, max(values) * 1.4]),
        yaxis=dict(autorange="reversed", color=fg, tickfont=dict(size=11)),
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=fg),
        height=max(280, len(items) * 28 + 60),
        margin=dict(l=10, r=80, t=40, b=10),
    )
    return fig

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
analysis_results = {}

def build_results_html(show_tags, dark_mode):
    if not analysis_results:
        color = "rgba(220,220,255,0.4)" if dark_mode else "rgba(0,0,0,0.3)"
        return f"<p style='color:{color};font-size:0.85rem;padding:8px'>Analyze files to see results.</p>"
    tag_color  = "#a5b4fc" if dark_mode else "#4f46e5"
    tag_bg     = "rgba(99,102,241,0.2)"  if dark_mode else "rgba(99,102,241,0.08)"
    tag_border = "rgba(99,102,241,0.35)" if dark_mode else "rgba(99,102,241,0.2)"
    row_bg     = "rgba(255,255,255,0.05)" if dark_mode else "rgba(0,0,0,0.025)"
    row_border = "rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.07)"
    fname_col  = "rgba(200,200,230,0.9)" if dark_mode else "rgba(0,0,0,0.7)"
    text_col   = "#e0e0f0" if dark_mode else "#1a1a2e"
    rows = []
    for fname, r in analysis_results.items():
        tag_html = ""
        if show_tags and r.get("tags"):
            tag_html = (
                f'<div style="display:flex;flex-wrap:wrap;gap:4px;padding:4px 10px 8px 10px">'
                + "".join(
                    f'<span style="background:{tag_bg};border:1px solid {tag_border};'
                    f'border-radius:20px;padding:2px 10px;font-size:0.75rem;color:{tag_color};text-transform:capitalize">{t}</span>'
                    for t in r["tags"]
                )
                + "</div>"
            )
        rows.append(
            f'<div style="padding:5px 10px;border-bottom:1px solid {row_border};margin-bottom:0">'
            f'<div style="font-size:0.8rem;color:{fname_col};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;margin-bottom:2px">{fname}</div>'
            f'<span style="font-size:0.95rem;color:{text_col}"><b>{r["main"]}</b> '
            f'<span style="opacity:0.8;font-size:0.85rem">({r["main_pct"]:.0f}%)</span>'
            f' · <b>{r["sub"]}</b> '
            f'<span style="opacity:0.8;font-size:0.85rem">({r["sub_pct"]:.0f}%)</span>'
            f'</span>'
            f'</div>'
            f'{tag_html}'
        )
    return '<div style="display:flex;flex-direction:column">' + "\n".join(rows) + "</div>"

def analyze_files(files, top_n, dark_mode, show_tags):
    global analysis_results
    if not files:
        empty = build_chart({}, "", top_n, dark_mode)
        return empty, empty, build_results_html(show_tags, dark_mode), gr.Dropdown(choices=["All files"], value="All files")
    analysis_results = {}
    for file in files:
        path  = file.name if hasattr(file, "name") else file
        fname = os.path.basename(path)
        try:
            style_scores, upper_scores = classify_genre(path)
            best_style = max(style_scores.items(), key=lambda x: x[1])
            best_upper = max(upper_scores.items(), key=lambda x: x[1]) if upper_scores else ("Unknown", 0)
            subgenre   = best_style[0].split("---")[-1] if "---" in best_style[0] else best_style[0]
            top_subs   = [k.split("---")[-1] for k in list(style_scores.keys())[:5]]
            tags       = get_tags(best_upper[0], top_subs, style_scores)
            analysis_results[fname] = {
                "styles": style_scores, "upper": upper_scores, "tags": tags,
                "main": best_upper[0], "sub": subgenre,
                "main_pct": best_upper[1] * 100, "sub_pct": best_style[1] * 100,
            }
        except Exception as e:
            msg = str(e)
            if "Cannot read audio" in msg:
                short = "File cannot be read — non-standard or corrupt MP3"
            else:
                short = msg[:80]
            print(f"ERROR processing {fname}: {msg}")
            analysis_results[fname] = {
                "styles": {}, "upper": {}, "tags": [],
                "main": "⚠ Unreadable", "sub": short, "main_pct": 0, "sub_pct": 0,
            }
    all_styles, all_upper = {}, {}
    for r in analysis_results.values():
        for g, s in r["styles"].items():
            all_styles[g] = all_styles.get(g, 0) + s
        for g, s in r["upper"].items():
            all_upper[g] = all_upper.get(g, 0) + s
    n = max(len(analysis_results), 1)
    all_styles = {k: v / n for k, v in all_styles.items()}
    all_upper  = {k: v / n for k, v in all_upper.items()}
    fig_upper = build_chart(all_upper,  f"Main Genres — {n} file(s)", top_n, dark_mode)
    fig_sub   = build_chart(all_styles, f"Subgenres — {n} file(s)",   top_n, dark_mode)
    choices   = ["All files"] + list(analysis_results.keys())
    return fig_upper, fig_sub, build_results_html(show_tags, dark_mode), gr.Dropdown(choices=choices, value="All files")

def update_charts(selected, top_n, dark_mode):
    if not analysis_results:
        empty = build_chart({}, "", top_n, dark_mode)
        return empty, empty
    if not selected or selected == "All files":
        all_styles, all_upper = {}, {}
        for r in analysis_results.values():
            for g, s in r["styles"].items():
                all_styles[g] = all_styles.get(g, 0) + s
            for g, s in r["upper"].items():
                all_upper[g] = all_upper.get(g, 0) + s
        n = max(len(analysis_results), 1)
        all_styles = {k: v / n for k, v in all_styles.items()}
        all_upper  = {k: v / n for k, v in all_upper.items()}
        suffix = f"{n} file(s)"
    else:
        r = analysis_results.get(selected, {})
        all_styles = r.get("styles", {})
        all_upper  = r.get("upper",  {})
        suffix = selected
    return (
        build_chart(all_upper,  f"Main Genres — {suffix}", top_n, dark_mode),
        build_chart(all_styles, f"Subgenres — {suffix}",   top_n, dark_mode),
    )

def update_tags(show_tags, dark_mode):
    return build_results_html(show_tags, dark_mode)

def on_theme_change(dark_mode, top_n, selected, show_tags):
    fu, fs = update_charts(selected, top_n, dark_mode)
    html   = build_results_html(show_tags, dark_mode)
    return fu, fs, html

def on_clear(dark_mode, show_tags):
    global analysis_results
    analysis_results = {}
    empty = build_chart({}, "", 10, dark_mode)
    return (
        None,
        build_results_html(show_tags, dark_mode),
        empty, empty,
        gr.Dropdown(choices=["All files"], value="All files"),
    )

# ---------------------------------------------------------------------------
# CSS — minimal, stable
# ---------------------------------------------------------------------------
CSS = """
*, *::before, *::after { box-sizing: border-box; }

html, body {
    background: #ffffff;
    color: #1a1a2e;
    transition: background 0.2s, color 0.2s;
}
html.dark, body.dark {
    background: #0f0f1a !important;
    color: #e0e0f0 !important;
}
body.dark .gradio-container,
body.dark .gr-box,
body.dark .gr-form,
body.dark .block,
body.dark .wrap,
body.dark .border-b,
body.dark .tabitem,
body.dark .upload-container,
body.dark .file-preview-holder,
body.dark table,
body.dark thead,
body.dark tbody,
body.dark tr,
body.dark th,
body.dark td {
    background: #1a1a2e !important;
    border-color: #2a2a4a !important;
    color: #e0e0f0 !important;
}
body.dark input,
body.dark select,
body.dark textarea {
    background: #1a1a2e !important;
    color: #e0e0f0 !important;
    border-color: #2a2a4a !important;
}
body.dark label,
body.dark span,
body.dark p,
body.dark .label-wrap,
body.dark .file-name,
body.dark .file-size {
    color: #e0e0f0 !important;
}
body.dark li[role=option] {
    background: #1a1a2e !important;
    color: #e0e0f0 !important;
}
body.dark input[type=checkbox] {
    accent-color: #818cf8 !important;
    width: 16px !important;
    height: 16px !important;
    border: 2px solid #818cf8 !important;
    outline: none !important;
    filter: invert(0.15) brightness(1.8) !important;
}
body.dark input[type=checkbox]:checked {
    filter: none !important;
    accent-color: #818cf8 !important;
    background: #818cf8 !important;
}
body.dark .gr-checkbox label,
body.dark input[type=checkbox] + span,
body.dark .checkbox-wrap span,
body.dark .checkbox-wrap label {
    color: #e0e0f0 !important;
    font-size: 0.95rem !important;
}
body.dark button:not(.analyze-btn) {
    background: #2a2a4a !important;
    color: #e0e0f0 !important;
    border-color: #3a3a6a !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
    background: transparent !important;
    font-size: 1rem !important;
}
/* Hide German upload text, show English */
.upload-container .upload-text::after {
    content: "Drop files here — or click to upload" !important;
}
.upload-container .upload-text {
    font-size: 0 !important;
}
/* Match genre box padding to file list */
#sgd-results .block {
    padding: 0 !important;
}

/* THE FIX: enforce exact 50/50 split at all times */
#sgd-two-col {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 20px !important;
    width: 100% !important;
    align-items: start !important;
}
#sgd-two-col > * {
    width: 100% !important;
    min-width: 0 !important;
    overflow: hidden !important;
}
#sgd-dropzone {
    min-height: 300px;
}
#sgd-results {
    min-height: 300px;
}

/* Buttons */
.analyze-btn {
    background: #2563eb !important;
    border: none !important;
    color: white !important;
    height: 52px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.analyze-btn:hover { background: #1d4ed8 !important; }
.clear-btn {
    height: 52px !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    background: #e5e7eb !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
}
.clear-btn:hover { background: #d1d5db !important; }
body.dark .clear-btn {
    background: #374151 !important;
    color: #e0e0f0 !important;
    border: 1px solid #4b5563 !important;
}
body.dark .clear-btn:hover { background: #4b5563 !important; }

.col-label {
    font-size: 0.75rem;
    font-weight: 700;
    opacity: 0.45;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.sgd-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
}
.sgd-toggle input[type=checkbox] {
    display: none !important;
}
.sgd-toggle-slider {
    position: relative;
    width: 40px;
    height: 22px;
    background: #d1d5db;
    border-radius: 11px;
    transition: background 0.2s;
    flex-shrink: 0;
}
.sgd-knob {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    display: block;
}
.sgd-toggle-label {
    font-size: 0.92rem;
    color: #1a1a2e;
}
body.dark .sgd-toggle-label {
    color: #e0e0f0 !important;
}
body.dark .sgd-toggle-slider {
    background: #374151;
}

.sgd-footer {
    text-align: center;
    font-size: 1rem;
    opacity: 0.65;
    margin-top: 24px;
    padding-top: 12px;
    border-top: 1px solid rgba(128,128,200,0.15);
}

/* Hide Gradio Footer  */
footer, .gradio-footer, .footer {
    display: none !important;
}

/* Hide Gradio API and Settings Buttons */
button:has(svg[aria-label="API"]),
button:has(svg[aria-label="Settings"]),
button[aria-label*="API"],
button[aria-label*="Settings"] {
    display: none !important;
}

.gradio-container {
    padding-bottom: 50px !important;
}
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="SGD Portable", css=CSS) as app:

    # Footer
    gr.HTML('<div style="position:fixed; bottom:0; left:0; width:100%; text-align:center; padding:8px; background:#1a1a2e; color:#a0a0b0; border-top:1px solid #2a2a4a; z-index:9999;">SGD Portable v1.0 — © 2026 by Reiner Prokein • made with ❤️ | MIT License</div>')

    logo_base64 = get_logo_base64()

    # Header
    with gr.Row():
        with gr.Column(scale=3):
            if logo_base64:
                logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height:64px;width:auto;">'
            else:
                logo_html = '<span style="font-size:1.5rem">🎧</span>'

            gr.HTML(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:16px 0 4px 0">
                {logo_html}
                <div>
                    <div style="font-size:1.35rem;font-weight:700">SGD Portable</div>
                    <div style="font-size:1rem;opacity:0.7;margin-top:2px">Song Genre Detector</div>
                    <div style="font-size:0.8rem;opacity:0.5;margin-top:3px">⚙ {DEVICE_LABEL}</div>
                </div>
            </div>
            """)
        with gr.Column(scale=1):
            dark_mode = gr.Checkbox(value=True,  visible=False)
            show_tags = gr.Checkbox(value=False, visible=False)
            with gr.Row():
                btn_dark = gr.Button("🌙 Dark theme: ON",  variant="primary", scale=1, size="sm")
                btn_tags = gr.Button("🏷 Tag cloud: OFF", variant="secondary", scale=1, size="sm")


    # Two-column block using elem_id for stable CSS targeting
    with gr.Row(elem_id="sgd-two-col"):
        with gr.Column(elem_id="sgd-dropzone"):
            gr.HTML('<div style="width:100%;min-width:550px;height:2px;visibility:hidden"></div>')
            gr.HTML('<div class="col-label">Dropzone</div>')
            file_input = gr.File(
                label="",
                file_types=["audio"],
                file_count="multiple",
            )
        with gr.Column(elem_id="sgd-results"):
            gr.HTML('<div style="width:100%;min-width:550px;height:2px;visibility:hidden"></div>')
            gr.HTML('<div class="col-label">Genre · Subgenre · Tags</div>')
            results_html = gr.HTML(
                "<p style='opacity:0.3;font-size:0.85rem;padding:8px'>Analyze files to see results.</p>"
            )

    # Buttons
    with gr.Row():
        btn       = gr.Button("▶  Analyze", elem_classes=["analyze-btn"], scale=5)
        clear_btn = gr.Button("✕  Clear",   elem_classes=["clear-btn"],   scale=1)

    # Chart controls
    with gr.Row():
        song_selector = gr.Dropdown(
            choices=["All files"], value="All files",
            label="Chart view", interactive=True, scale=4,
        )
        top_n = gr.Dropdown(
            choices=list(range(5, 55, 5)), value=10,
            label="Show top", scale=1,
        )

    # Charts
    with gr.Row():
        chart_upper = gr.Plot(show_label=False)
        chart_sub   = gr.Plot(show_label=False)

    gr.HTML('<div class="sgd-footer">The results are based on probabilities. AI makes mistakes. Please always double-check.</div>')

    # ---------------------------------------------------------------------------
    # Toggle button states
    # ---------------------------------------------------------------------------
    def toggle_dark(current):
        new_val = not current
        label = "🌙 Dark theme: ON" if new_val else "☀ Dark theme: OFF"
        variant = "primary" if new_val else "secondary"
        return new_val, gr.Button(label, variant=variant)

    def toggle_tags(current):
        new_val = not current
        label = "🏷 Tag cloud: ON" if new_val else "🏷 Tag cloud: OFF"
        variant = "primary" if new_val else "secondary"
        return new_val, gr.Button(label, variant=variant)

    btn_dark.click(
        fn=toggle_dark,
        inputs=[dark_mode],
        outputs=[dark_mode, btn_dark],
    )
    dark_mode.change(
        fn=None, inputs=[dark_mode], outputs=[],
        js="""(dark) => {
            if (dark) {
                document.documentElement.classList.add('dark');
                document.body.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
                document.body.classList.remove('dark');
            }
        }"""
    )
    dark_mode.change(
        fn=on_theme_change,
        inputs=[dark_mode, top_n, song_selector, show_tags],
        outputs=[chart_upper, chart_sub, results_html],
    )

    btn_tags.click(
        fn=toggle_tags,
        inputs=[show_tags],
        outputs=[show_tags, btn_tags],
    )
    btn_tags.click(
        fn=update_tags,
        inputs=[show_tags, dark_mode],
        outputs=[results_html],
    )

    # ---------------------------------------------------------------------------
    # Events
    # ---------------------------------------------------------------------------


    file_input.change(
        fn=None, inputs=[], outputs=[],
        js="""() => {
            setTimeout(() => {
                const clearBtn = document.querySelector('.clear-btn');
                if (!clearBtn) return;
                const rows = document.querySelectorAll('.file-preview tbody tr');
                if (rows.length > 0) {
                    clearBtn.style.setProperty('background', '#f97316', 'important');
                    clearBtn.style.setProperty('color', 'white', 'important');
                    clearBtn.style.setProperty('border', 'none', 'important');
                } else {
                    clearBtn.style.removeProperty('background');
                    clearBtn.style.removeProperty('color');
                    clearBtn.style.removeProperty('border');
                }
            }, 300);
        }"""
    )

    btn.click(
        fn=analyze_files,
        inputs=[file_input, top_n, dark_mode, show_tags],
        outputs=[chart_upper, chart_sub, results_html, song_selector],
    )

    clear_btn.click(
        fn=on_clear,
        inputs=[dark_mode, show_tags],
        outputs=[file_input, results_html, chart_upper, chart_sub, song_selector],
    )

    song_selector.change(
        fn=update_charts,
        inputs=[song_selector, top_n, dark_mode],
        outputs=[chart_upper, chart_sub],
    )

    top_n.change(
        fn=update_charts,
        inputs=[song_selector, top_n, dark_mode],
        outputs=[chart_upper, chart_sub],
    )



    app.load(
        fn=None, inputs=[], outputs=[],
        js="""() => {
            document.documentElement.classList.add('dark');
            document.body.classList.add('dark');
            function fixDropzone() {
                document.querySelectorAll('.upload-text, .wrap span').forEach(el => {
                    if (el.textContent.includes('ablegen') || el.textContent.includes('Hochladen') || el.textContent.includes('oder')) {
                        const p = el.closest('.upload-container');
                        if (p) {
                            const existing = p.querySelector('.sgd-drop-hint');
                            if (!existing) {
                                const hint = document.createElement('div');
                                hint.className = 'sgd-drop-hint';
                                hint.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;pointer-events:none;font-size:0.95rem;opacity:0.5';
                                hint.innerHTML = '⬆ Drop files here<br><small>or click to upload</small>';
                                p.style.position = 'relative';
                                p.appendChild(hint);
                            }
                        }
                    }
                });
            }
            setTimeout(fixDropzone, 500);
        }"""
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)
