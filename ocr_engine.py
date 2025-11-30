import easyocr
import numpy as np
def loade_ocr_model(languges=['en','fa'],wieghts_path=None):
    reader = easyocr.Reader(languges)
    print("easy OCR Model loaded")
    return reader

def predict_ocr(reader,image_path):
    result = reader.readtext(image_path,detail=0)
    full_text = " ".join(result)
    return full_text