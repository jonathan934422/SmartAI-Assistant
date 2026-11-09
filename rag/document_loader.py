from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    """Extract text from all pages of a PDF."""

    reader = PdfReader(pdf_file)

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n\n".join(pages_text)


def split_text_into_chunks(
    text,
    chunk_size=1000,
    chunk_overlap=200,
):
    """Split document text into overlapping chunks."""

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks