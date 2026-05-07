from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "paper" / "figures"
ANALYSIS_DIR = ROOT / "paper" / "analysis" / "c3a_ci_cost_20260506"
NUMBERS_PATH = ROOT / "paper" / "data" / "numbers.yaml"


COLORS = {
    "ink": "#20262e",
    "muted": "#5f6c7b",
    "grid": "#d7dee8",
    "panel": "#f6f8fb",
    "blue": "#315f9d",
    "blue_dark": "#1f3f68",
    "green": "#2f7f67",
    "orange": "#bd6f2a",
    "red": "#b04b4b",
    "purple": "#6b5fb5",
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.edgecolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_verified_numbers(path: Path = NUMBERS_PATH) -> dict[str, float | list[float]]:
    """Small parser for the flat numeric leaves used by paper/data/numbers.yaml."""
    wanted: dict[str, float | list[float]] = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.startswith('"') or not value:
                continue
            if value.startswith("[") and value.endswith("]"):
                try:
                    wanted[key.strip()] = [float(part.strip()) for part in value[1:-1].split(",")]
                except ValueError:
                    pass
                continue
            try:
                wanted[key.strip()] = float(value)
            except ValueError:
                continue
    return wanted


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def short_model(name: str) -> str:
    repl = {
        "allenai_wildguard": "WildGuard",
        "meta-llama_Llama-Guard-3-8B": "LlamaGuard 3",
        "meta-llama_Llama-Guard-4-12B": "LlamaGuard 4",
        "Qwen_Qwen3Guard-Gen-0.6B": "Qwen3Guard",
        "google_shieldgemma-2b": "ShieldGemma",
        "mistralai_Ministral-8B-Instruct-2410": "Ministral-8B",
        "microsoft_phi-4": "Phi-4",
        "google_gemma-3-12b-it": "Gemma-3-12B",
        "gemini-2.0-flash": "Gemini 2.0 Flash",
        "GPT-OSS-20B": "GPT-OSS-20B",
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o mini",
        "Qwen_Qwen2.5-14B-Instruct": "Qwen2.5-14B",
        "Qwen_Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "Qwen_Qwen3-8B": "Qwen3-8B",
        "meta-llama_Llama-3.1-8B-Instruct": "Llama3.1-8B",
    }
    return repl.get(name, name.replace("_", " "))
