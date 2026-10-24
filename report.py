"""Turn results/<host>-<date>/*.json into one markdown table: python report.py results/pi5-20261024"""
import json
import sys
from pathlib import Path


def table(rows, cols):
    rows = [r for r in rows if any(c in r for c in cols)]
    if not rows:
        return ""
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main(folder):
    folder = Path(folder)
    parts = [f"# wonder-bench: {folder.name}\n"]
    for f in sorted(folder.glob("*.json")):
        d = json.loads(f.read_text())
        if not isinstance(d, dict) or f.name.endswith(".raw.json"):  # raw tool output; its summary has its own file
            continue
        parts.append(f"## {f.stem}\n")
        if "sysinfo" in d:
            s = d["sysinfo"]
            parts.append(f"{s.get('model')} · {s.get('mem_gb')} GB · kernel {s.get('kernel')} · governor {s.get('governor')} · {s.get('time')}\n")
        rows = d.get("rows", [])
        if rows:
            cols = list(dict.fromkeys(k for r in rows for k in r if not isinstance(r[k], (dict, list))))
            parts.append(table(rows, cols) + "\n")
        for k, v in d.items():
            if k not in ("rows", "sysinfo"):
                parts.append(f"- **{k}**: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
        parts.append("")
    md = "\n".join(parts)
    (folder / "REPORT.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sorted(Path("results").iterdir())[-1])
