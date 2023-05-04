import io
import csv
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import nltk
nltk.download('punkt')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    resource_manager = PDFResourceManager()
    buffer = io.StringIO()
    laparams = LAParams()
    device = TextConverter(resource_manager, buffer, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, device)
    for page in PDFPage.get_pages(pdf_file):
        interpreter.process_page(page)
    text = buffer.getvalue()
    device.close()
    buffer.close()
    return text

# Open the PDF file
pdf_file = open('/Users/adilqaisar/Documents/Zacks/AAPL ERR.pdf', 'rb')

# Extract the text from the PDF file
text = extract_text_from_pdf(pdf_file)

# Replace newlines with spaces
text = text.replace('\n', ' ')

# Split the text into sentences
sentences = nltk.sent_tokenize(text)

# Write the sentences to a CSV file with three sentences in one column
with open('AAPLEER.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["sentences", "countofwords"])
    sentence_count = 0
    row = ''
    for sentence in sentences:
        row += sentence + ' '
        sentence_count += 1
        if sentence_count == 3:
            word_count = len(row.split())
            writer.writerow([row, word_count])
            row = ''
            sentence_count = 0
    if sentence_count != 0:
        word_count = len(row.split())
        writer.writerow([row, word_count])
