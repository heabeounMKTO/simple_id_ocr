import os
import cv2
from id_extractor import IdExtractor
from id_structs import IdKeypoints
from pprint import pprint
from id_utils import perspective_transform_from_kpts, perspective_transform_from_kpts_nocrop
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--folder", type=str, help="target folder")
    parser.add_argument("--img", type=str, help="DA IMAGE")
    opts = parser.parse_args()
    img = cv2.imread(opts.img)
    custom_config = r"-l khm --oem 3 --psm 6"
    extractor = IdExtractor("./models/id_kpt_vn.pt", "./models/id_ki.pt")
    # _results = extractor.extract_end2end(img, debug=True)
    keypoints = extractor.get_keypoints(img)
    translated_info = perspective_transform_from_kpts(keypoints, img)
    cv2.imwrite("debug_crops/translated.png", translated_info)

