from tika import parser
import glob

files = glob.glob("*.pdf")

for file in files:
    if file == "5.Land Acquisition Act 2013.pdf": #OCR only file
        continue
    # Parse data from file
    file_data = parser.from_file(file)
    # Get files text content
    text = file_data['content']
    f = open(f"{file[:-4]}.txt", "w+")
    f.write(text)
    f.close()