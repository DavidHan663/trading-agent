"""Scan tracked files without printing credential values. Run before publishing."""

import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "API credential": re.compile(r"(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "populated env secret": re.compile(r"^[ \t]*[A-Z][A-Z0-9_]*(?:API_KEY|PASSWORD|TOKEN)[ \t]*=[ \t]*([^\r\n#]+)", re.MULTILINE),
}


def findings(path):
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeError, OSError):
        return []
    hits = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(content):
            if label == "populated env secret":
                value = match.group(1).strip().strip("\"'")
                if (not value or "..." in value
                        or value.startswith(("your_", "YOUR_", "替换", "${"))):
                    continue
            hits.append((content.count("\n", 0, match.start()) + 1, label))
    return hits


def main():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            capture_output=True, check=True)
    count = 0
    for name in sorted(set(result.stdout.decode().split("\0"))):
        if not name:
            continue
        for line, label in findings(Path(name)):
            print(f"{name}:{line}: possible {label} (value hidden)")
            count += 1
    print(f"Secret scan: {count} potential findings. This is a heuristic, not a guarantee.")
    return bool(count)


if __name__ == "__main__":
    sys.exit(main())
