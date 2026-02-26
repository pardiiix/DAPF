from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
import pandas as pd


# ---------- helpers ----------
def norm_id(x: str) -> str:
    x = str(x).strip()
    m = re.search(r"(S\d+)", x)
    return m.group(1) if m else x


def ensure_unzipped(zip_path: Path, extract_dir: Path) -> Path:
    """
    Unzip zip_path into extract_dir (if not already unzipped).
    Returns the folder that contains 'transcription/'.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    # If already extracted and contains transcription/, reuse
    if (extract_dir / "transcription").exists():
        return extract_dir

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Sometimes zips contain a top-level folder; search for transcription/
    if (extract_dir / "transcription").exists():
        return extract_dir

    candidates = list(extract_dir.rglob("transcription"))
    if not candidates:
        raise FileNotFoundError(f"Unzipped {zip_path} but couldn't find a 'transcription/' folder under {extract_dir}")
    return candidates[0].parent


# ---------- CHAT cleaning ----------
def clean_chat_text(raw: str, keep_par: bool = True, keep_inv: bool = False) -> str:
    if raw is None:
        return ""
    raw = str(raw)
    if not raw.strip():
        return ""

    kept = []
    current_keep = False  # whether we are currently inside a kept speaker utterance

    for ln in raw.splitlines():
        ln = ln.rstrip()

        # Drop headers and annotation tiers
        if ln.startswith("@") or ln.startswith("%"):
            current_keep = False
            continue

        # Speaker tiers
        m = re.match(r"^\*([A-Z]{3}):\s*(.*)$", ln)
        if m:
            spk, utt = m.group(1), m.group(2)
            if spk == "PAR" and keep_par:
                kept.append(utt)
                current_keep = True
            elif spk == "INV" and keep_inv:
                kept.append(utt)
                current_keep = True
            else:
                current_keep = False
            continue

        # Continuation lines: keep ONLY if they follow a kept speaker tier
        if (ln.startswith("\t") or ln.startswith(" ")) and current_keep:
            kept.append(ln.strip())

    text = " ".join(kept)

    # Remove weird control chars / timecodes (common in CHAT exports)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)          # control chars
    text = re.sub(r"\d+_\d+", " ", text)                  # timecodes like 0_3957
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- metadata parsing ----------
def read_meta_semicolon(path: Path, has_mmse: bool) -> pd.DataFrame:
    """
    Train: ID ; age ; gender ; mmse
    Test:  ID ; age ; gender
    """
    df = pd.read_csv(path, sep=r"\s*;\s*", engine="python", header=None)
    if has_mmse:
        df.columns = ["id", "age", "gender", "mmse"]
    else:
        df.columns = ["id", "age", "gender"]
    df["id"] = df["id"].map(norm_id)
    for c in ["age", "gender"]:
        df[c] = df[c].astype(str).str.strip()
    if has_mmse:
        df["mmse"] = df["mmse"].astype(str).str.strip()
    return df


# ---------- transcripts ----------
def read_transcripts(folder: Path, label: str | None) -> list[dict]:
    rows = []
    for fp in sorted(folder.glob("*")):
        if fp.is_dir():
            continue
        sid = norm_id(fp.stem)
        raw = fp.read_text(errors="ignore")
        rows.append({"filename": sid, "text_raw": raw, "label": label})
    return rows


def build_from_transcription_root(trans_root: Path, domain_value: str, keep_inv: bool) -> pd.DataFrame:
    cc_dir = trans_root / "cc"
    cd_dir = trans_root / "cd"

    rows = []
    if cc_dir.exists() and cd_dir.exists():
        rows += read_transcripts(cc_dir, "ct")
        rows += read_transcripts(cd_dir, "AD")
    else:
        # unlabeled test style: all files directly under transcription/
        rows += read_transcripts(trans_root, None)

    df = pd.DataFrame(rows)
    df["text"] = df["text_raw"].apply(lambda x: clean_chat_text(x, keep_par=True, keep_inv=keep_inv))

    df["domain"] = domain_value
    df["domain1"] = domain_value
    df["domain2"] = domain_value

    df = df.drop_duplicates("filename").reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", required=True, help="e.g., data/train")
    p.add_argument("--test_root", required=True, help="e.g., data/test")
    p.add_argument("--out_dir", required=True, help="e.g., data/")
    p.add_argument("--domain_value", default="picture description")
    p.add_argument("--keep_inv", action="store_true", help="include interviewer (*INV:) lines too")
    args = p.parse_args()

    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- TRAIN --------
    train_zip = train_root / "transcription.zip"
    if not train_zip.exists():
        raise FileNotFoundError(f"Expected {train_zip} based on your folder screenshot.")

    train_extract = train_root / "transcription_extracted"
    train_base = ensure_unzipped(train_zip, train_extract)
    train_trans_root = train_base / "transcription"

    train_df = build_from_transcription_root(train_trans_root, args.domain_value, keep_inv=args.keep_inv)

    # attach train metadata (cc/cd meta files)
    cc_meta = train_root / "cc_meta_data.txt"
    cd_meta = train_root / "cd_meta_data.txt"
    meta_frames = []
    if cc_meta.exists():
        m = read_meta_semicolon(cc_meta, has_mmse=True)
        m["label"] = "ct"
        meta_frames.append(m)
    if cd_meta.exists():
        m = read_meta_semicolon(cd_meta, has_mmse=True)
        m["label"] = "AD"
        meta_frames.append(m)

    if meta_frames:
        meta = pd.concat(meta_frames, ignore_index=True).rename(columns={"id": "filename"})
        train_df = train_df.merge(meta, on=["filename", "label"], how="left")

    print("[train] rows:", len(train_df), "labels:", train_df["label"].value_counts(dropna=False).to_dict())
    print("[train] empty text ratio:", (train_df["text"].fillna("").str.strip() == "").mean())

    # -------- TEST --------
    # Support either:
    #  - test/transcription.zip
    #  - test/transcription/...
    test_trans_root = None
    test_zip = test_root / "transcription.zip"
    if test_zip.exists():
        test_extract = test_root / "transcription_extracted"
        test_base = ensure_unzipped(test_zip, test_extract)
        test_trans_root = test_base / "transcription"
    else:
        # maybe already a folder
        if (test_root / "transcription").exists():
            test_trans_root = test_root / "transcription"
        else:
            raise FileNotFoundError(f"Expected either {test_zip} or {test_root/'transcription'} for test.")

    test_df = build_from_transcription_root(test_trans_root, args.domain_value, keep_inv=args.keep_inv)

    # attach test metadata if present
    test_meta = test_root / "meta_data.txt"
    if test_meta.exists():
        m = read_meta_semicolon(test_meta, has_mmse=False).rename(columns={"id": "filename"})
        test_df = test_df.merge(m, on="filename", how="left")

    print("[test] rows:", len(test_df), "labels:", test_df["label"].value_counts(dropna=False).to_dict())
    print("[test] empty text ratio:", (test_df["text"].fillna("").str.strip() == "").mean())

    # -------- WRITE --------
    train_out = out_dir / "adress-train_all.csv"
    train_df.to_csv(train_out, index=False)
    print("Wrote:", train_out.resolve())

    # Always write the name the training code expects
    test_out = out_dir / "adress-test_all.csv"

    # If unlabeled, set a placeholder label so downstream code expecting the column won't break
    if test_df["label"].isna().all():
        test_df["label"] = "unknown"

    test_df.to_csv(test_out, index=False)
    print("Wrote:", test_out.resolve())


if __name__ == "__main__":
    main()
