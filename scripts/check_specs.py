#!/usr/bin/env python3
"""Check service specifications without third-party packages or model data."""

import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SERVICES = {"Backend": "BE", "MLService": "ML", "iOSApp": "IOS"}
REQUIRED = {"docs/README.md", "docs/spec/README.md", "MLService/experiments/template/README.md"}
REQUIRED.update(f"docs/spec/{part}/README.md" for part in (
    "constitution", "conventions", "glossary", "integration", "templates",
))
for service in SERVICES:
    REQUIRED.add(f"{service}/docs/ai/README.md")
    REQUIRED.add(f"{service}/docs/spec/README.md")
    REQUIRED.update(f"{service}/docs/spec/{part}/README.md" for part in (
        "nonfunctional", "integration", "open-questions", "traceability", "patches",
    ))
ID = r"(?:TC-PATCH|FR|NFR|TC|INT|GAP|OQ|PATCH)-(?:BE|ML|IOS|MP)-[A-Z0-9]+(?:-[A-Z0-9]+)*"
DEFINITION = re.compile(rf"^#{{2,6}} +({ID})(?=\s|$)", re.MULTILINE)
REFERENCE = re.compile(rf"\b({ID})\b")
LINK = re.compile(r"\[[^\]]*\]\((<[^>]+>|[^\s)]+)\)")


def prose(text):
    """Ignore examples inside Markdown fences, including template ID examples."""
    return re.sub(r"^(`{3,}|~{3,})[^\n]*\n.*?^\1\s*$", "", text, flags=re.MULTILINE | re.DOTALL)


def anchors(text):
    explicit = re.findall(r'<a\s+id="([^"]+)"\s*>', text)
    result = set(explicit)
    counts = {}
    for heading in re.findall(r"^#{1,6} +(.+)$", text, re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        result.add(f"{slug}-{count}" if count else slug)
        counts[slug] = count + 1
    return result, explicit


def managed(name):
    return (
        name in {"docs/README.md", "scripts/README.md", "MLService/experiments/template/README.md"}
        or name.startswith("docs/spec/")
        or any(name.startswith(f"{s}/docs/{part}/") for s in SERVICES for part in ("spec", "ai"))
    )


def check_documents(root, paths):
    paths = set(paths)
    errors = [f"{name}: required document missing or ignored" for name in sorted(REQUIRED - paths)]
    documents = {}
    for name in sorted(paths):
        path = Path(name)
        if path.suffix.lower() != ".md":
            continue
        if path.name != "README.md" and name != "AGENTS.md":
            errors.append(f"{name}: Markdown must be README.md (root AGENTS.md excepted)")
        if managed(name):
            if (root / name).is_file():
                documents[name] = prose((root / name).read_text(encoding="utf-8"))
            else:
                errors.append(f"{name}: document missing on disk")
    for service in SERVICES:
        if not any(name.startswith(f"{service}/docs/spec/functional/") for name in documents):
            errors.append(f"{service}: functional specification missing")

    definitions = {}
    anchor_map = {}
    for name, text in documents.items():
        available, explicit = anchors(text)
        anchor_map[name] = available
        if len(explicit) != len(set(explicit)):
            errors.append(f"{name}: duplicate explicit anchor")
        for match in DEFINITION.finditer(text):
            identity = match.group(1)
            if identity in definitions:
                errors.append(f"{name}: duplicate ID {identity} (also {definitions[identity]})")
            definitions[identity] = name
            if identity.lower() not in explicit:
                errors.append(f"{name}: ID {identity} needs explicit lowercase anchor")
            service = name.split("/", 1)[0]
            owner = identity.split("-")[2 if identity.startswith("TC-PATCH-") else 1]
            if service in SERVICES and owner != SERVICES[service]:
                errors.append(f"{name}: ID {identity} belongs to another component")

    for name, text in documents.items():
        # Only service contract/evidence files contain normative ID references.
        if name.split("/", 1)[0] in SERVICES and "/docs/" in name:
            for identity in set(REFERENCE.findall(text)):
                if identity not in definitions:
                    errors.append(f"{name}: undefined ID {identity}")
        for match in DEFINITION.finditer(text):
            identity = match.group(1)
            if not identity.startswith(("FR-", "NFR-", "PATCH-")):
                continue
            section = re.split(r"^#{1,6} ", text[match.end():], maxsplit=1, flags=re.MULTILINE)[0]
            cases = {ref for ref in REFERENCE.findall(section) if ref.startswith("TC-")}
            if not cases:
                errors.append(f"{name}: {identity} has no acceptance case")
            for case in cases:
                if definitions.get(case) != name:
                    errors.append(f"{name}: {case} must be defined beside its rule")

        # ponytail: inline links only; add a Markdown parser if richer syntax is adopted.
        for target in LINK.findall(text):
            url = urlsplit(target.strip("<>"))
            if url.scheme or url.netloc:
                continue
            destination = (root / name).parent / unquote(url.path) if url.path else root / name
            try:
                relative = destination.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                relative = ""
            if relative not in paths or not destination.is_file():
                errors.append(f"{name}: local link missing, ignored or outside repo: {target}")
                continue
            if url.fragment and destination.suffix == ".md":
                if relative not in anchor_map:
                    anchor_map[relative] = anchors(prose(destination.read_text(encoding="utf-8")))[0]
                if unquote(url.fragment) not in anchor_map[relative]:
                    errors.append(f"{name}: unknown anchor in {target}")
    return errors


def self_test():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        fixture = {name: "# Document\n" for name in REQUIRED}
        for service in SERVICES:
            fixture[f"{service}/docs/spec/functional/example/README.md"] = "# Example\n"
        sample = "Backend/docs/spec/functional/example/README.md"
        fixture[sample] += (
            '<a id="fr-be-example"></a>\n### FR-BE-EXAMPLE — Rule\n'
            'Acceptance: [TC-BE-EXAMPLE](#tc-be-example).\n'
            '<a id="tc-be-example"></a>\n### TC-BE-EXAMPLE — Case\n'
            'Given input, when processed, then expected outcome.\n'
        )
        for name, text in fixture.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        assert not check_documents(root, fixture), check_documents(root, fixture)
        cases = (
            ("docs/spec/extra.md", "# Wrong name\n", "Markdown must"),
            (sample, fixture[sample] + "[bad](missing.py)\n", "local link"),
            (sample, fixture[sample] + "[bad](#absent)\n", "unknown anchor"),
            ("Backend/docs/spec/duplicate/README.md", fixture[sample], "duplicate ID"),
            (sample, fixture[sample].replace("Acceptance: [TC-BE-EXAMPLE](#tc-be-example).", "None."), "no acceptance"),
            (sample, fixture[sample] + "TC-BE-UNDEFINED\n", "undefined ID"),
            (sample, fixture[sample].replace('<a id="fr-be-example"></a>', ""), "explicit lowercase"),
            (sample, fixture[sample].replace("FR-BE-", "FR-IOS-BE-"), "another component"),
        )
        for name, text, expected in cases:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            diagnostics = check_documents(root, set(fixture) | {name})
            assert any(expected in error for error in diagnostics), diagnostics
            if name in fixture:
                path.write_text(fixture[name], encoding="utf-8")
            else:
                path.unlink()
        assert any("required document" in error for error in check_documents(root, set(fixture) - {"docs/README.md"}))
        ignored = root / "private.txt"
        ignored.write_text("Present but not part of the checkout inventory")
        (root / sample).write_text(fixture[sample] + "[ignored](../../../../../private.txt)\n")
        assert any("local link" in error for error in check_documents(root, fixture))
        (root / sample).write_text(fixture[sample])
        assert not check_documents(root, fixture)
    print("Specification self-test OK: valid fixture and 10 rejection cases")


def main():
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if sys.argv[1:]:
        print("Usage: python3 scripts/check_specs.py [--self-test]", file=sys.stderr)
        return 2
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
        )
        paths = {name for name in output.decode("utf-8").split("\0") if name}
        errors = check_documents(ROOT, paths)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Specification check failed: {error}", file=sys.stderr)
        return 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Specification check OK: service structure, Markdown, links, anchors, IDs and acceptance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
