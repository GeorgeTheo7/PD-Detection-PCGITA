"""Registry of the main model and all ablation variants.

Each entry: which frozen SSL model, which fusion, and which acoustic branches are used.
"""

EXPERIMENTS = {
    # Main configuration
    "coattn_xlsr53":   dict(ssl="xlsr53",    fusion="coattn"),
    # SSL backbone ablation (co-attention, all three branches)
    "coattn_hubert":   dict(ssl="hubert",    fusion="coattn"),
    "coattn_w2v2base": dict(ssl="w2v2_base", fusion="coattn"),
    # Fusion ablation (HuBERT)
    "concat_hubert":   dict(ssl="hubert", fusion="concat"),
    "mlb_hubert":      dict(ssl="hubert", fusion="mlb"),
    "mfb_hubert":      dict(ssl="hubert", fusion="mfb"),
    # Modality ablation (HuBERT)
    "spec_hubert":     dict(ssl="hubert", fusion="coattn", use_mfcc=False),
    "mfcc_hubert":     dict(ssl="hubert", fusion="coattn", use_spec=False),
    "hubert_only":     dict(ssl="hubert", fusion="ssl_only", use_spec=False, use_mfcc=False),
    "spec_mfcc":       dict(ssl=None,     fusion="no_ssl"),
}

DISPLAY_NAMES = {
    "coattn_xlsr53": "Co-Attention + wav2vec2 XLSR-53", "coattn_hubert": "Co-Attention + HuBERT Base",
    "coattn_w2v2base": "Co-Attention + wav2vec2 Base", "concat_hubert": "Concatenation (HuBERT)",
    "mlb_hubert": "MLB (HuBERT)", "mfb_hubert": "MFB (HuBERT)", "spec_hubert": "Spectrogram + HuBERT",
    "mfcc_hubert": "MFCC + HuBERT", "hubert_only": "HuBERT only",
    "spec_mfcc": "Spectrogram + MFCC (no SSL)",
}

# Groups used to build the ablation tables (reference model listed last)
ABLATION_TABLES = {
    "SSL backbone (co-attention, all branches)": ["coattn_w2v2base", "coattn_xlsr53", "coattn_hubert"],
    "Fusion method (HuBERT)": ["concat_hubert", "mlb_hubert", "mfb_hubert", "coattn_hubert"],
    "Input modalities (HuBERT)": ["hubert_only", "spec_mfcc", "mfcc_hubert", "spec_hubert", "coattn_hubert"],
}
ABLATION_REFERENCE = "coattn_hubert"


def get(name):
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment '{name}'. Available: {', '.join(EXPERIMENTS)}")
    cfg = dict(EXPERIMENTS[name])
    cfg.setdefault("use_spec", True)
    cfg.setdefault("use_mfcc", True)
    return cfg
