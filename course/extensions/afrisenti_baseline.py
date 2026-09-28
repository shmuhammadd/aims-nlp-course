"""Character TF-IDF baseline for permitted local AfriSenti-style TSV exports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def load(path):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"id", "text", "label", "language"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns {sorted(required)}")
        rows = list(reader)
    if not rows or any(not all(r.get(k) for k in required) for r in rows):
        raise ValueError(f"{path}: rows must have nonempty required fields")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate IDs")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for split in ["train", "dev", "test"]:
        p.add_argument("--"+split, required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/afrisenti-results.json"))
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output file")
    rows = {s: load(getattr(args, s)) for s in ["train", "dev", "test"]}
    for a,b in [("train","dev"),("train","test"),("dev","test")]:
        # Exact checks supplement, but do not replace, source-group split auditing.
        if {r["id"] for r in rows[a]} & {r["id"] for r in rows[b]}:
            p.error(f"ID overlap between {a} and {b}")
        hashes = lambda part: {hashlib.sha256(r["text"].strip().casefold().encode()).hexdigest() for r in part}
        if hashes(rows[a]) & hashes(rows[b]):
            p.error(f"Exact normalized text overlap between {a} and {b}")
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    import sklearn
    labels = sorted({r["label"] for r in rows["train"]})
    if len(labels) < 2:
        p.error("Training needs at least two classes")
    if any(r["label"] not in labels for s in ["dev","test"] for r in rows[s]):
        p.error("Evaluation contains an unseen label; check label schema")
    x = lambda s: [r["text"] for r in rows[s]]
    y = lambda s: [r["label"] for r in rows[s]]
    candidates=[]
    for c in [.1, 1., 10.]:
        model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2,5), min_df=1, max_features=50000),
                              LogisticRegression(C=c, max_iter=1000, random_state=42))
        model.fit(x("train"),y("train"))
        score=f1_score(y("dev"),model.predict(x("dev")),labels=labels,average="macro",zero_division=0)
        candidates.append((score,c,model))
    score,c,model=max(candidates,key=lambda z:z[0])
    predictions=model.predict(x("test"))
    metrics={"accuracy":accuracy_score(y("test"),predictions),"macro_f1":f1_score(y("test"),predictions,labels=labels,average="macro",zero_division=0)}
    slices={}
    for language in sorted({r["language"] for r in rows["test"]}):
        idx=[i for i,r in enumerate(rows["test"]) if r["language"]==language]
        target=[y("test")[i] for i in idx]; pred=[predictions[i] for i in idx]
        slices[language]={"n":len(idx),"accuracy":accuracy_score(target,pred),"macro_f1":f1_score(target,pred,labels=labels,average="macro",zero_division=0)}
    result={"selected_C":c,"dev_macro_f1":score,"test":metrics,"language_slices":slices,"labels":labels,
            "sklearn":sklearn.__version__,"seed":42,"split_sha256":{s:hashlib.sha256(getattr(args,s).read_bytes()).hexdigest() for s in rows},
            "predictions":[{"id":r["id"],"target":r["label"],"prediction":str(pred)} for r,pred in zip(rows["test"],predictions)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f: json.dump(result,f,indent=2)
    print(json.dumps({"selected_C":c,"dev_macro_f1":score,"test":metrics,"language_slices":slices},indent=2))


if __name__ == "__main__":
    main()
