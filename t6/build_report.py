"""Build the technical report as a self-contained HTML file and a PDF.

HTML: Python-Markdown with a contents list, print-friendly CSS and every image embedded.
PDF: the same HTML printed by headless Edge or Chrome (whichever is installed).

Run with `uv run python -m t6.build_report`. Writes reports/technical_report.html and .pdf.
"""
import base64
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

from t6 import config as C

SRC = C.REPORTS / "technical_report.md"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "msedge", "chrome", "google-chrome", "chromium",
]

CSS = """
:root { --ink:#0b0b0b; --muted:#52514e; --line:#e1e0d9; --soft:#f4f3ef; --blue:#2a78d6; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif; color: var(--ink);
       background: #fcfcfb; line-height: 1.55; font-size: 11pt; max-width: 860px; margin: 0 auto;
       padding: 28px 22px 60px; }
h1 { font-size: 1.7em; line-height: 1.25; margin: 0 0 .4em; }
h2 { font-size: 1.3em; margin: 1.8em 0 .5em; padding-bottom: .2em; border-bottom: 2px solid var(--line); }
h3 { font-size: 1.1em; margin: 1.4em 0 .4em; }
h2, h3 { break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: var(--blue); }
blockquote { margin: 1em 0; padding: .1em 1em; border-left: 4px solid var(--blue); background: var(--soft);
             color: var(--muted); }
code { font-family: Consolas, "Courier New", monospace; font-size: .9em; background: var(--soft);
       padding: .05em .3em; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .86em; break-inside: avoid; }
th, td { border: 1px solid var(--line); padding: 5px 8px; text-align: left; vertical-align: top;
         overflow-wrap: anywhere; }
th { background: var(--soft); font-weight: 600; }
th:first-child, td:first-child { min-width: 9em; }
img { display: block; max-width: 100%; max-height: 150mm; width: auto; height: auto; object-fit: contain;
      margin: 1em auto .3em; }
p:has(> img) { break-after: avoid; break-inside: avoid; margin-bottom: 0; }
p > em:only-child { display: block; color: var(--muted); font-size: .9em; margin-bottom: 1.2em; }
hr { border: 0; border-top: 1px solid var(--line); margin: 1.4em 0; }
nav.toc { background: var(--soft); border: 1px solid var(--line); border-radius: 6px; padding: .6em 1.2em;
          margin: 1em 0 1.4em; font-size: .95em; }
nav.toc ol { margin: .3em 0; padding-left: 0; list-style: none; }
nav.toc strong { display: block; margin-top: .2em; }
@page { size: A4; margin: 16mm 14mm; }
@media print { body { background: #fff; max-width: none; padding: 0; font-size: 10pt; } a { color: inherit; } }
"""


def embed_images(html: str, base: Path) -> str:
    def swap(m):
        src = m.group(2)
        if src.startswith(("http", "data:")):
            return m.group(0)
        path = (base / src).resolve()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        return f'{m.group(1)}src="data:{mime};base64,{data}"'
    return re.sub(r'(<img [^>]*?)src="([^"]+)"', swap, html)


def contents(tokens) -> str:
    items = "".join(f'<li><a href="#{t["id"]}">{t["name"]}</a></li>' for t in tokens if t["level"] == 2)
    return f'<nav class="toc"><strong>Contents</strong><ol>{items}</ol></nav>'


def build_html() -> Path:
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"])
    body = md.convert(SRC.read_text(encoding="utf-8"))
    title = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S).group(1)
    toc = contents([c for t in md.toc_tokens for c in [t, *t["children"]]])
    body = body.replace("<hr />", "<hr />\n" + toc, 1)  # contents after the header block
    body = embed_images(body, SRC.parent)
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{re.sub('<[^>]+>', '', title)}</title><style>{CSS}</style></head><body>{body}</body></html>")
    out = SRC.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out


def find_browser() -> str | None:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    return None


def build_pdf(html: Path) -> Path | None:
    browser = find_browser()
    if browser is None:
        print("No Edge or Chrome found: HTML written, PDF skipped.")
        return None
    pdf = html.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as profile:  # own profile: works while a browser is open
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={pdf}", html.as_uri()],
                       check=True, timeout=180, capture_output=True)
    return pdf


def main() -> None:
    html = build_html()
    print(f"{html.name}: {html.stat().st_size / 1e6:.1f} MB")
    pdf = build_pdf(html)
    if pdf and pdf.exists():
        pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
        print(f"{pdf.name}: {pdf.stat().st_size / 1e6:.1f} MB, {pages} pages")


if __name__ == "__main__":
    sys.exit(main())
