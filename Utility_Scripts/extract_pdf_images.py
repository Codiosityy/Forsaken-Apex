import fitz  # PyMuPDF
import os
import re

# ---------------- CONFIG ----------------
INPUT_DIR = "pdfs"
OUTPUT_DIR = "figures"

FIGURE_REGEX = re.compile(
    r'\b(fig\.?|figure)\s*(\d+)\s*(?:[\(\[]?([a-z])[\)\]]?)?',
    re.IGNORECASE
)

MAX_VERTICAL_DISTANCE = 250     # px
MIN_IMAGE_AREA = 10_000         # filter small icons
BBOX_TOLERANCE = 5

# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def image_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bboxes_overlap(b1, b2, tol=BBOX_TOLERANCE):
    return not (
        b1[2] < b2[0] - tol or
        b1[0] > b2[2] + tol or
        b1[3] < b2[1] - tol or
        b1[1] > b2[3] + tol
    )


def bbox_above(img_bbox, text_bbox):
    return img_bbox[3] <= text_bbox[1]


for pdf_file in os.listdir(INPUT_DIR):
    if not pdf_file.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(INPUT_DIR, pdf_file)
    pdf_name = os.path.splitext(pdf_file)[0]
    pdf_out = os.path.join(OUTPUT_DIR, pdf_name)
    os.makedirs(pdf_out, exist_ok=True)

    doc = fitz.open(pdf_path)
    seen_xrefs = set()

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        page_images = page.get_images(full=True)

        text_blocks = []
        image_blocks = []

        for b in blocks:
            if b["type"] == 0:  # text
                text_blocks.append(b)
            elif b["type"] == 1:  # image
                if image_area(b["bbox"]) >= MIN_IMAGE_AREA:
                    image_blocks.append(b)

        # ---------- CAPTION DETECTION ----------
        for tb in text_blocks:
            text = ""
            for line in tb.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "")

            match = FIGURE_REGEX.search(text)
            if not match:
                continue

            fig_num = match.group(2)
            caption_bbox = tb["bbox"]

            # ---------- FIND IMAGES ABOVE ----------
            candidates = []
            for ib in image_blocks:
                if not bbox_above(ib["bbox"], caption_bbox):
                    continue

                vertical_dist = caption_bbox[1] - ib["bbox"][3]
                if vertical_dist <= MAX_VERTICAL_DISTANCE:
                    candidates.append(ib)

            if not candidates:
                continue

            candidates.sort(key=lambda b: b["bbox"][0])

            # ---------- MATCH LAYOUT → REAL IMAGE ----------
            for idx, img_block in enumerate(candidates):
                matched_xref = None

                for img in page_images:
                    xref = img[0]
                    try:
                        img_bbox = page.get_image_bbox(img)
                    except Exception:
                        continue

                    if bboxes_overlap(img_bbox, img_block["bbox"]):
                        matched_xref = xref
                        break

                if matched_xref is None or matched_xref in seen_xrefs:
                    continue

                seen_xrefs.add(matched_xref)

                base_image = doc.extract_image(matched_xref)
                ext = base_image["ext"]
                img_bytes = base_image["image"]

                suffix = chr(ord('a') + idx) if len(candidates) > 1 else ""
                name = f"Figure_{fig_num}{suffix}.{ext}"

                with open(os.path.join(pdf_out, name), "wb") as f:
                    f.write(img_bytes)

    doc.close()

print("✅ Figure extraction complete.")
