from pathlib import Path
from pdf2image import convert_from_path
import cv2


def process_page(pdf_path, page_index, image_path_page):
    pdf_page = convert_from_path(
        pdf_path,
        dpi=400,
        first_page=page_index+1,
        last_page=page_index+1
    )[0]
    save(pdf_page, image_path_page)
    to_sharp(image_path_page)
    del pdf_page

def save(pdf_page, image_path_page):
    pdf_page.save(image_path_page, "PNG")

def to_sharp(image_path_page):
    image = cv2.imread(str(image_path_page))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    clean = cv2.fastNlMeansDenoising(binary, h=30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    sharpened = cv2.dilate(clean, kernel, iterations=1)
    sharpened = cv2.erode(sharpened, kernel, iterations=1)

    output_path = Path(image_path_page)
    cv2.imwrite(output_path, sharpened)
    return sharpened

def preprocess(pdf_path, page_index, image_path="/tmp/floor_plan.png"):
    image_path = Path(image_path)
    image_path_page = image_path.parent.joinpath(image_path.stem).with_suffix(f".{str(page_index).zfill(2)}{image_path.suffix}")
    process_page(pdf_path, page_index, image_path_page)

    return image_path_page