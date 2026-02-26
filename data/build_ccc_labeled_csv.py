from __future__ import annotations
import argparse, re, zipfile
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET

def norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^(mr|mrs|ms|dr)\.?\s+", "", s)   # drop titles
    s = re.sub(r"\s+", " ", s)
    return s

def norm_stem(fn: str) -> str:
    # "Willard_Shaf_001.txt" -> "Willard_Shaf_001"
    fn = str(fn).strip()
    fn = fn.rsplit("/", 1)[-1]
    for ext in [".trs", ".txt", ".cha"]:
        if fn.lower().endswith(ext):
            fn = fn[:-len(ext)]
    return fn

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_text_from_trs(xml_bytes: bytes) -> str:
    # Transcriber XML (.trs). Decode leniently.
    xml_str = xml_bytes.decode("ISO-8859-1", errors="ignore")
    root = ET.fromstring(xml_str)

    chunks = []
    for turn in root.findall(".//Turn"):
        # Collect visible text (ignore tags but keep tail text)
        parts = []
        if turn.text:
            parts.append(turn.text)
        for node in turn.iter():
            if node is turn:
                continue
            if node.tail:
                parts.append(node.tail)
        t = clean_text(" ".join(parts))
        if t:
            chunks.append(t)
    return clean_text(" ".join(chunks))

def label_from_condition(cond: str) -> str:
    c = (cond or "").strip().lower()
    return "AD" if "alz" in c else "ct"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts_zip", required=True)
    ap.add_argument("--participants_csv", required=True)
    ap.add_argument("--transcripts_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--domain_value", default="health and wellbeing")
    args = ap.parse_args()

    # --- load participants (labels) ---
    p = pd.read_csv(args.participants_csv, dtype=str).fillna("")
    p = p[p["participant_role"].str.lower().str.strip() == "interviewee"].copy()
    p["pname_norm"] = p["participant"].apply(norm_name)
    p["label"] = p["participant_condition_or_disease"].apply(label_from_condition)

    # If duplicates (same interviewee appears multiple times), keep "AD" over ct/unknown
    label_rank = {"AD": 2, "ct": 1, "unknown": 0}
    p["rank"] = p["label"].map(label_rank)
    p = p.sort_values("rank", ascending=False).drop_duplicates("pname_norm", keep="first")

    # --- load transcript mapping ---
    tmap = pd.read_csv(args.transcripts_csv, dtype=str).fillna("")
    tmap["stem"] = tmap["transcript"].apply(norm_stem)
    tmap["iname_norm"] = tmap["transcript_creator_interviewee"].apply(norm_name)

    # join transcript->label via interviewee name
    tmap = tmap.merge(
        p[["pname_norm", "label", "participant_gender", "participant_age_range", "participant"]],
        left_on="iname_norm",
        right_on="pname_norm",
        how="left"
    )

    # --- read transcript text from zip ---
    zpath = Path(args.transcripts_zip)
    with zipfile.ZipFile(zpath, "r") as z:
        namelist = z.namelist()
        trs_files = [n for n in namelist if n.lower().endswith(".trs")]

        # map stem->zip member
        zip_index = {norm_stem(n): n for n in trs_files}

        rows = []
        for _, r in tmap.iterrows():
            stem = r["stem"]
            if stem not in zip_index:
                continue
            xml_bytes = z.read(zip_index[stem])
            text = extract_text_from_trs(xml_bytes)
            if not text.strip():
                continue

            label = r.get("label", "")
            if label == "" or pd.isna(label):
                label = "unknown"

            rows.append({
                "filename": stem,
                "text": text,
                "label": label,
                "domain": args.domain_value,
                "domain1": args.domain_value,
                "domain2": args.domain_value,
                "age_range": r.get("participant_age_range", ""),
                "gender": r.get("participant_gender", ""),
                "interviewee": r.get("participant", r.get("transcript_creator_interviewee", "")),
                "interviewer": r.get("transcript_interviewer", ""),
                "transcript_type": r.get("transcript_type", ""),
            })

    out = pd.DataFrame(rows).drop_duplicates("filename").reset_index(drop=True)

    print("Total CCC transcripts with text:", len(out))
    print("Label counts:", out["label"].value_counts(dropna=False).to_dict())

    # Optional: drop unknowns for training
    train = out[out["label"].isin(["ct", "AD"])].copy()
    print("Usable for training (ct/AD only):", len(train))
    print("Train label counts:", train["label"].value_counts().to_dict())

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_csv, index=False)
    print("Wrote:", Path(args.out_csv).resolve())

if __name__ == "__main__":
    main()