"""Generate a simple PDF from the project's DELIVERY_EXPLANATION.md.

Requires: reportlab

Usage:
    python3 scripts/create_pdf.py

The script writes `Voice_converter_explanation.pdf` to the repo root.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "DELIVERY_EXPLANATION.md"
OUT = ROOT / "Voice_converter_explanation.pdf"


def md_to_paragraphs(text: str):
    paragraphs = []
    for block in text.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        paragraphs.append(block)
    return paragraphs


def build_pdf():
    if not MD.exists():
        raise SystemExit(f"Markdown file not found: {MD}")

    text = MD.read_text(encoding="utf-8")
    para_texts = md_to_paragraphs(text)

    doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    heading = ParagraphStyle('Heading', parent=styles['Heading2'], spaceAfter=6)

    story = []
    for p in para_texts:
        if p.startswith('# '):
            story.append(Paragraph(p[2:].strip(), styles['Heading1']))
        elif p.startswith('## '):
            story.append(Paragraph(p[3:].strip(), styles['Heading2']))
        elif p.startswith('### '):
            story.append(Paragraph(p[4:].strip(), styles['Heading3']))
        else:
            # replace backticks with simple <b> tags for inline code appearance
            p = p.replace('`', '')
            p = p.replace('---', '')
            story.append(Paragraph(p.replace('\n', '<br/>'), normal))
        story.append(Spacer(1, 8))

    doc.build(story)
    print(f"Wrote PDF: {OUT}")


if __name__ == '__main__':
    build_pdf()
