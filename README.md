# simple_id_ocr
fast and simple OCR for id cards 

## installing dependencies


first, install tesseract for khmer language. <br>
#### on linux 
`apt install tesseract-ocr-khm`


then, install the python3 dependencies

`pip install -r requirements.txt`

download the models from the releases page 
|model name| what it does|
|---|---|
|id_ki| keyinformation extractor|
|id_kpt| four corners keypoints for warp transform|

and then place it in the `./models` directory.

then , you can start the webui with `python3 webui.py`
