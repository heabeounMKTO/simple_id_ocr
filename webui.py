import gradio as gr 
import cv2
from numpy import extract
from id_extractor import IdExtractor 
from id_structs import IdKeypoints 
from id_utils import perspective_transform_from_kpts
import re

extractor = IdExtractor("./models/id_kpt.pt", "./models/id_ki.pt")

def remove_addr_prefix(text):
    """Remove everything before and including the first colon (:)."""
    return re.sub(r"^[^:]*: *", "", text)

def format_dictionary(data):
    """Format dictionary into a neat string with line breaks between fields."""
    formatted = ""
    for key, values in data.items():
        formatted += f"{key}:\n" + "\n".join(values).strip() + "\n\n"
    return formatted.strip()

def run_ocr(image):
    results = extractor.extract_end2end(image, debug=True)
    try:
        del results["bottom"]
    except Exception as _e:
        pass
    print(results)
    # final_addr_string = "".join(filter(None, results["address"]))
    # for idx, i in enumerate(results["address"]):
        # print(remove_addr_prefix(i))
        # if "អាសយដ្ឋានៈ" in i:
        #     results["address"][idx] = results["address"][idx].replace("អាសយដ្ឋានៈ", "") 
        
    # print(results["address"])
    # results["address"] = remove_addr_prefix(final_addr_string)
    return format_dictionary(results)

demo = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Textbox(label="Extracted Information"),
    title="Id Information Extractor",
    description="Upload an image of a cambodian id on the left to extract text on the right."
)

# Launch the application
demo.launch()

