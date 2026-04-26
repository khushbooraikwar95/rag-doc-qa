from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_path = Path(__file__).parent / "sample_data.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()
# print(docs[3])

