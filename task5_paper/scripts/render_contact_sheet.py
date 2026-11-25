import sys
from pathlib import Path

import pymupdf
from PIL import Image

source = Path(sys.argv[1])
output = Path(sys.argv[2])
doc = pymupdf.open(source)
pages = [page.get_pixmap(matrix=pymupdf.Matrix(0.75, 0.75), alpha=False).pil_image() for page in doc]
width = max(page.width for page in pages)
height = max(page.height for page in pages)
sheet = Image.new("RGB", (2 * width + 24, 3 * height + 36), "white")
for index, page in enumerate(pages):
    sheet.paste(page, ((index % 2) * (width + 12), (index // 2) * (height + 12)))
sheet.save(output)
print(f"Rendered {len(pages)} pages to {output}")
