"""Build lecture decks on the Imperial template from course/slides/src/*.md.

Usage:  python scripts/build_slides.py [--only 11_steering]
Needs:  pip install -r requirements-slides.txt

Source format (one file per lecture). Slides are separated by a line holding only '---'.
Each slide starts with 'key: value' header lines, then a blank line, then the body.

  type:      title | section | big | content (default) | closing
  title:     slide title (state the claim or the question, not the topic)
  subtitle:  optional second line
  layout:    split (text left, visual right) | stack (text above visual); chosen automatically if absent
  footer:    title/closing slides only (author | date line)

Body, in any order:
  plain line            paragraph
  ### text              bold subheading
  - text                bullet;  two leading spaces = sub-bullet
  **bold**  `code`      inline styles
  ![](figures/x.png)    picture (path relative to course/slides/)
  $$ latex $$           display equation (matplotlib mathtext); several lines allowed between $$ ... $$
  ``` ... ```           code block
  | a | b |             table (first row is the header)
  Notes: ...            everything after this line becomes speaker notes

The build fails if text does not fit its box at 16 pt: split the slide instead of shrinking it.
"""
import argparse
import copy
import hashlib
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "course/slides"
TEMPLATE = SLIDES / "template/imperial_template.potx"
BLUE, VIOLET, INK, WHITE = RGBColor(0x00, 0x00, 0xCD), RGBColor(0x7B, 0x68, 0xEE), RGBColor(0, 0, 0), RGBColor(255, 255, 255)
LEFT, TOP, WIDTH, BOTTOM, GAP = 0.36, 1.53, 12.62, 6.85, 0.4
LAYOUT = {"title": "Title Slide Smoke", "section": "Section Header Blue", "big": "Large Text Blue",
          "content": "Title and Content Smoke", "closing": "Closing Slide Blue"}
FONT_DIRS = ["/System/Library/Fonts/Supplemental", "/Library/Fonts", "C:/Windows/Fonts", "/usr/share/fonts/truetype/msttcorefonts"]


def font(size, bold=False, mono=False):
    names = (["Courier New Bold.ttf", "courbd.ttf"] if bold else ["Courier New.ttf", "cour.ttf"]) if mono else \
        (["Arial Bold.ttf", "arialbd.ttf"] if bold else ["Arial.ttf", "arial.ttf"])
    for d in FONT_DIRS:
        for n in names:
            if (Path(d) / n).exists():
                return ImageFont.truetype(str(Path(d) / n), int(size * 10))  # 10 px per pt for precision
    raise SystemExit("Arial not found: text fitting cannot be checked on this machine")


def runs(text):
    """Split inline **bold** and `code` into (text, bold, mono) runs."""
    out = []
    for part in re.split(r"(\*\*.+?\*\*|`.+?`)", text):
        if part.startswith("**") and part.endswith("**"):
            out.append((part[2:-2], True, False))
        elif part.startswith("`") and part.endswith("`"):
            out.append((part[1:-1], False, True))
        elif part:
            out.append((part, False, False))
    return out


def text_height(paras, width_in, size):
    """Height in inches of wrapped paragraphs [(level, text)] at a point size."""
    total = 0.0
    for level, text in paras:
        indent = {0: 0, 1: 0, 2: 0.25, 3: 0.5}[level]
        avail = (width_in - indent - 0.2) * 720  # px at 10 px/pt
        line, lines = 0.0, 1
        for chunk, bold, mono in runs(text):
            f = font(size, bold or level == 1, mono)
            for word in re.findall(r"\S+\s*", chunk):
                w = f.getlength(word)
                if line + f.getlength(word.rstrip()) > avail and line > 0:
                    lines, line = lines + 1, 0.0
                line += w
        total += lines * size * 1.2 / 72 + size * 0.45 / 72
    return total * 1.08  # safety margin for renderer differences


def equation_png(lines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    key = hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]
    out = SLIDES / "figures/eq" / f"{key}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.rcParams["mathtext.fontset"] = "cm"
        fig = plt.figure(figsize=(0.1, 0.1))
        for i, line in enumerate(lines):
            fig.text(0, -i * 1.0, f"${line}$", fontsize=30, va="top")
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08, transparent=True)
        plt.close(fig)
    return out


def parse(path):
    slides = []
    for block in re.split(r"^---\s*$", path.read_text(), flags=re.M):
        if not block.strip():
            continue
        head, _, body = block.strip("\n").partition("\n\n")
        meta = {}
        for line in head.splitlines():
            m = re.match(r"(type|title|subtitle|layout|footer):\s*(.*)", line)
            if not m:
                raise SystemExit(f"{path.name}: bad header line {line!r} (missing blank line before the body?)")
            meta[m.group(1)] = m.group(2).strip()
        body, _, notes = body.partition("\nNotes:")
        if body.startswith("Notes:"):
            body, notes = "", body[6:]
        paras, visuals, lines, i = [], [], body.splitlines(), 0
        while i < len(lines):
            s = lines[i]
            if s.startswith("```"):
                j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("```"))
                visuals.append(("code", lines[i + 1:j])); i = j
            elif s.strip().startswith("$$"):
                text = s.strip()[2:]
                while not text.rstrip().endswith("$$"):
                    i += 1; text += "\n" + lines[i]
                visuals.append(("eq", [x.strip() for x in text.rstrip()[:-2].splitlines() if x.strip()]))
            elif s.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                        rows.append(cells)
                    i += 1
                visuals.append(("table", rows)); i -= 1
            elif m := re.match(r"!\[.*?\]\((.+?)\)", s):
                visuals.append(("image", m.group(1)))
            elif s.startswith("### "):
                paras.append((1, s[4:]))
            elif s.startswith("  - "):
                paras.append((3, s[4:]))
            elif s.startswith("- "):
                paras.append((2, s[2:]))
            elif s.strip():
                paras.append((0, s.strip()))
            i += 1
        first = next((l for l in lines if l.strip()), "")
        meta["visual_first"] = first.startswith(("```", "|", "![")) or first.strip().startswith("$$")
        slides.append((meta, paras, visuals, notes.strip()))
    return slides


def fill(shape, text):
    shape.text_frame.paragraphs[0].text = ""
    r = shape.text_frame.paragraphs[0].add_run(); r.text = text


def write_paras(frame, paras, size):
    frame.word_wrap = True
    for n, (level, text) in enumerate(paras):
        p = frame.paragraphs[0] if n == 0 else frame.add_paragraph()
        p.level = level
        for chunk, bold, mono in runs(text):
            r = p.add_run(); r.text = chunk; r.font.size = Pt(size)
            if bold or level == 1:
                r.font.bold = True
            if mono:
                r.font.name = "Courier New"


def add_visual(slide, kind, data, x, y, w, h, where):
    if kind in ("image", "eq"):
        src = equation_png(data) if kind == "eq" else SLIDES / data
        if not src.exists():
            raise SystemExit(f"{where}: missing picture {data}")
        iw, ih = Image.open(src).size
        scale = min(w / iw, h / ih, (1 / 200 if kind == "eq" else 1e9))  # equations keep their natural size if it fits
        pw, ph = iw * scale, ih * scale
        slide.shapes.add_picture(str(src), Inches(x + (w - pw) / 2), Inches(y), Inches(pw), Inches(ph))
        return ph
    if kind == "code":
        size = 16
        longest = max((len(l) for l in data), default=1)
        while size > 12 and (longest * size * 0.6 / 72 > w - 0.4 or len(data) * size * 1.2 / 72 > h - 0.3):
            size -= 1
        if longest * size * 0.6 / 72 > w - 0.4 or len(data) * size * 1.2 / 72 > h - 0.3:
            raise SystemExit(f"{where}: code block too large ({len(data)} lines, {longest} columns); shorten it")
        hh = len(data) * size * 1.2 / 72 + 0.3
        box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(hh))
        box.fill.solid(); box.fill.fore_color.rgb = WHITE; box.line.fill.background(); box.shadow.inherit = False
        tf = box.text_frame; tf.word_wrap = False
        tf.margin_left = tf.margin_right = Inches(0.2); tf.margin_top = tf.margin_bottom = Inches(0.15)
        for n, line in enumerate(data):
            p = tf.paragraphs[0] if n == 0 else tf.add_paragraph()
            p.alignment = 1
            r = p.add_run(); r.text = line or " "; r.font.name = "Courier New"; r.font.size = Pt(size); r.font.color.rgb = INK
        return hh
    rows, cols = len(data), len(data[0])
    size = 16 if rows <= 7 else 14
    col_w = w / cols
    heights = [max(text_height([(0, c)], col_w, size) for c in row) / 1.08 - size * 0.45 / 72 + 0.16 for row in data]  # cells have no paragraph gap
    if sum(heights) > h:
        raise SystemExit(f"{where}: table needs {sum(heights):.1f} in but has {h:.1f}; cut rows or words")
    table = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(sum(heights))).table
    for r, row in enumerate(data):
        table.rows[r].height = Inches(heights[r])
        for c, text in enumerate(row):
            cell = table.cell(r, c); cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if r == 0 else WHITE
            cell.text_frame.paragraphs[0].text = ""
            for chunk, bold, mono in runs(text):
                run = cell.text_frame.paragraphs[0].add_run(); run.text = chunk; run.font.size = Pt(size)
                run.font.bold = bold or r == 0; run.font.color.rgb = WHITE if r == 0 else INK
                if mono:
                    run.font.name = "Courier New"
    return sum(heights)


def build(src):
    buf = io.BytesIO()
    with zipfile.ZipFile(TEMPLATE) as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():  # a .potx opens as a deck once its content type says so
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(b"presentationml.template.main+xml", b"presentationml.presentation.main+xml")
            zout.writestr(item, data)
    prs = Presentation(buf)
    ids = prs.slides._sldIdLst
    for sld in list(ids):  # drop the template's instruction slides
        prs.part.drop_rel(sld.rId); ids.remove(sld)
    layouts = {l.name: l for l in prs.slide_masters[0].slide_layouts}
    figs = src.with_name(src.stem + "_figures.py")
    if figs.exists():
        subprocess.run([sys.executable, str(figs)], cwd=SLIDES, check=True)
    for number, (meta, paras, visuals, notes) in enumerate(parse(src), 1):
        kind = meta.get("type", "content")
        where = f"{src.name} slide {number} ({meta.get('title', '')[:40]})"
        layout = layouts[LAYOUT[kind]]
        slide = prs.slides.add_slide(layout)
        ph = {p.placeholder_format.idx: p for p in slide.placeholders}
        fill(ph[0], meta.get("title", ""))
        if kind == "title" and len(meta.get("subtitle", "")) > 45:
            raise SystemExit(f"{where}: title-slide subtitle over 45 characters runs into the author line")
        if kind in ("title", "closing"):
            if 1 in ph:
                fill(ph[1], meta.get("subtitle", ""))
            fill(ph[10], meta.get("footer", ""))
        elif kind == "section":
            fill(ph[1], meta.get("subtitle", ""))
        elif kind == "content":
            if font(26, True).getlength(meta.get("title", "")) / 720 > WIDTH - 0.2:
                raise SystemExit(f"{where}: title is longer than one line")
            if meta.get("subtitle"):
                if font(26).getlength(meta["subtitle"]) / 720 > WIDTH - 0.2:
                    raise SystemExit(f"{where}: subtitle is longer than one line")
                fill(ph[13], meta["subtitle"])
            else:
                ph[13]._element.getparent().remove(ph[13]._element)
            for item in layout.placeholders:  # slide number and footer are not inherited automatically
                if item.placeholder_format.type in (15, 13):  # FOOTER, SLIDE_NUMBER
                    slide.shapes._spTree.append(copy.deepcopy(item._element))
            split = meta.get("layout", "split" if visuals and visuals[0][0] == "image" and paras else "stack") == "split" and visuals and paras
            text_w = (WIDTH - GAP) / 2 if split else WIDTH
            room = BOTTOM - TOP - (0 if split else 1.3 * len(visuals))  # each stacked visual keeps at least 1.3 in
            body, text_h = ph[1], 0
            if paras:
                size = next((s for s in (20, 18, 16) if text_height(paras, text_w, s) <= room), None)
                if size is None:
                    raise SystemExit(f"{where}: text does not fit at 16 pt; cut words or split the slide")
                text_h = text_height(paras, text_w, size)
                write_paras(body.text_frame, paras, size)
            else:
                body._element.getparent().remove(body._element)
            x, w = (LEFT + text_w + GAP, text_w) if split else (LEFT, WIDTH)
            visual_first = meta["visual_first"] and not split
            y = TOP if split or visual_first or not paras else TOP + text_h + 0.25
            bottom = BOTTOM - (text_h + 0.25 if visual_first and paras else 0)
            for n, (vkind, data) in enumerate(visuals):
                later = visuals[n + 1:]
                pictures_later = sum(k == "image" for k, _ in later)
                share = bottom - y - 1.3 * (len(later) - pictures_later)  # equations, code and tables take what they need
                if vkind == "image":
                    share /= pictures_later + 1                          # pictures split what is left
                elif pictures_later:
                    share = min(share, 2.5)
                if share < 0.6:
                    raise SystemExit(f"{where}: no room left for {vkind}; split the slide")
                y += add_visual(slide, vkind, data, x, y, w, share - 0.2, where) + 0.25
            if paras:
                body.left, body.width = Inches(LEFT), Inches(text_w)
                body.top = Inches(y if visual_first else TOP)
                body.height = Inches(BOTTOM - TOP if split else text_h)
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
    out = SLIDES / f"{src.stem}.pptx"
    prs.save(out)
    print(f"built {out.relative_to(ROOT)}: {len(prs.slides)} slides")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="lecture stem, such as 11_steering")
    args = parser.parse_args()
    sources = [p for p in sorted((SLIDES / "src").glob("[0-9x]*.md")) if not args.only or p.stem == args.only]
    if not sources:
        parser.error("no matching slide source")
    for source in sources:
        build(source)
