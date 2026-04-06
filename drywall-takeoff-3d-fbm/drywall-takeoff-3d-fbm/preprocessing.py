from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pdf2image.pdf2image import pdfinfo_from_path
from pdf2image import convert_from_path
import cv2
from concurrent.futures import ThreadPoolExecutor


def process_page(pdf_path, page_index, image_path_page, vector_pdf_page):
    vector_page = PdfReader(pdf_path).pages[page_index]
    pdf_page = convert_from_path(
        pdf_path,
        dpi=400,
        first_page=page_index+1,
        last_page=page_index+1
    )[0]
    save(pdf_page, vector_page, image_path_page, vector_pdf_page)
    to_sharp(image_path_page)
    del vector_page
    del pdf_page

def save(pdf_page, vector_page, image_path_page, vector_pdf_page):
    pdf_page.save(image_path_page, "PNG")
    writer = PdfWriter()
    writer.add_page(vector_page)
    vector_pdf_page.parent.mkdir(parents=True, exist_ok=True)
    with open(vector_pdf_page, "wb") as f:
        writer.write(f)

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

def preprocess(pdf_path, image_path="/tmp/floor_plan.png"):
    print("converting to image")
    #pages = convert_from_path(
    #    pdf_path,
    #    dpi=250,
    #)
    metadata = pdfinfo_from_path(pdf_path)
    n_pages = metadata["Pages"]
    print(n_pages)
    image_path_pages = list()
    vector_pdf_pages = list()
    image_path = Path(image_path)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = list()
        for page_index in range(n_pages):
            image_path_page = image_path.parent.joinpath(image_path.stem).with_suffix(f".{str(page_index).zfill(2)}{image_path.suffix}")
            vector_pdf_page = image_path.parent.joinpath(str(page_index).zfill(2)).joinpath(f"scaled_{image_path.stem}").with_suffix(".pdf")
            futures.append(
                executor.submit(
                    process_page,
                    pdf_path,
                    #pdf_page,
                    page_index,
                    image_path_page,
                    vector_pdf_page,
                )
            )
            image_path_pages.append(image_path_page)
            vector_pdf_pages.append(vector_pdf_page)
        [future.result() for future in futures]

    return vector_pdf_pages, image_path_pages