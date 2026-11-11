"""Step 1 - Preprocessing: select tasks, convert to mono 16 kHz, cut into 5 s chunks.

Input layout (original PC-GITA folders):
    <raw_dir>/read text/ayerfuialmedico/sin normalizar/{hc,pd}/*.wav
    <raw_dir>/monologue/sin normalizar/{hc,pd}/*.wav

Output:
    <out_dir>/chunks/<subject>_<task>_<i>.wav
    <out_dir>/pcgita_chunks.csv   (one row per chunk)
"""
import os
import re

import numpy as np
import pandas as pd

from . import config


def extract_subject_id(filename: str) -> str:
    """Return the speaker code (e.g. 'AVPEPUDEAC0005') from any PC-GITA filename.

    PC-GITA filenames are not consistent across tasks:
        AVPEPUDEAC0005_readtext.wav
        AVPEPUDEAC0005-Monologo-NR.wav
    so splitting on '_' is NOT safe (it silently creates two IDs per speaker).
    """
    m = re.match(config.SUBJECT_ID_REGEX, filename)
    if m is None:
        raise ValueError(f"Could not parse a PC-GITA subject ID from '{filename}'")
    return m.group(1)


def load_metadata(path):
    """Optional PC-GITA metadata spreadsheet -> {subject_id: {age, gender, updrs}}."""
    if path is None or not os.path.exists(path):
        return {}
    meta = pd.read_excel(path)
    meta.columns = [str(c).strip().upper() for c in meta.columns]
    id_col = "CODIGO" if "CODIGO" in meta.columns else meta.columns[0]
    out = {}
    for _, row in meta.iterrows():
        sid = str(row[id_col]).strip()
        out[sid] = {"age": row.get("AGE", np.nan), "gender": row.get("SEX", np.nan),
                    "updrs": row.get("UPDRS", np.nan)}
    return out


def chunk_signal(sig: np.ndarray):
    """Split a 1-D signal into 5 s non-overlapping chunks (zero-pad the last one, drop it if < 2.5 s)."""
    chunks = []
    n = int(np.ceil(len(sig) / config.CHUNK_SAMPLES))
    for i in range(n):
        c = sig[i * config.CHUNK_SAMPLES:(i + 1) * config.CHUNK_SAMPLES]
        if len(c) < int(config.MIN_VALID_SEC * config.TARGET_SR):
            continue
        if len(c) < config.CHUNK_SAMPLES:
            c = np.pad(c, (0, config.CHUNK_SAMPLES - len(c)))
        chunks.append(c)
    return chunks


def prepare_pcgita(raw_dir, out_dir, metadata_path=None):
    import librosa
    import soundfile as sf

    chunk_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    meta = load_metadata(metadata_path)
    rows = []

    for task, rel in config.TASKS.items():
        for cls, label in config.CLASS_DIRS.items():
            folder = os.path.join(raw_dir, rel, cls)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Missing folder: {folder}")
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))
            print(f"{task:<10} {cls.upper()}: {len(files)} recordings")
            for fname in files:
                sid = extract_subject_id(fname)
                sig, _ = librosa.load(os.path.join(folder, fname), sr=config.TARGET_SR, mono=True)
                for i, chunk in enumerate(chunk_signal(sig)):
                    out_name = f"{sid}_{task}_{i}.wav"
                    sf.write(os.path.join(chunk_dir, out_name), chunk, config.TARGET_SR)
                    rows.append({"filename": out_name, "subject_id": sid, "label": label, "task": task,
                                 "chunk_index": i, "original_source": fname, **meta.get(sid, {})})

    df = pd.DataFrame(rows)

    # Sanity checks: PC-GITA has 50 PD + 50 HC speakers, each with exactly one label
    assert df.groupby("subject_id")["label"].nunique().max() == 1, "A speaker has two labels"
    per_class = df.groupby("label")["subject_id"].nunique().to_dict()
    print(f"Chunks: {len(df)} | speakers per class: {per_class} (0 = HC, 1 = PD)")
    if per_class != {0: 50, 1: 50}:
        print("WARNING: expected 50 HC and 50 PD speakers - check the input folders.")

    csv_path = os.path.join(out_dir, "pcgita_chunks.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return csv_path
