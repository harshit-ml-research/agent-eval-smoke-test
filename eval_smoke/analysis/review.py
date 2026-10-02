"""Export saved trajectories to a local HTML file for manual review."""

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path


def export(inputs: list[Path], output: Path) -> int:
    cases = []
    statuses = Counter()
    for source in inputs:
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected an object")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            status = str(record.get("status", "unknown"))
            statuses[status] += 1
            label = record.get("scenario", record.get("task_id", f"line {line_number}"))
            score = record.get("score", record.get("root_cause_hit_proxy", "unscored"))
            summary = f"{source.name}: {label}, attempt {record.get('attempt', 1)}, {status}, score {score}"
            metadata = {key: value for key, value in record.items() if key not in {"trajectory", "messages", "answer"}}
            sections = [f"<details><summary>{escape(summary)}</summary>"]
            for title, value in [("Final answer", record.get("answer", "")), ("Metadata and score", metadata), ("Complete trajectory", record.get("trajectory", [])), ("Complete messages", record.get("messages", []))]:
                content = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
                sections.append(f"<h3>{title}</h3><pre>{escape(content)}</pre>")
            sections.append("</details>")
            cases.append("\n".join(sections))
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Evaluation review</title><style>
body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px system-ui;background:#faf9f6;color:#222}
details{border-top:1px solid #ccc;padding:16px 0}summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:16px;font-size:13px}
</style><h1>Evaluation review</h1><p>All cases are included. Scores are recorded outputs, not manual verdicts.</p>"""
    document += f"<p>{len(cases)} cases. {escape(str(dict(statuses)))}</p>" + "\n".join(cases) + "</html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(f"Exported {export(args.inputs, args.output)} cases to {args.output}")


if __name__ == "__main__":
    main()
