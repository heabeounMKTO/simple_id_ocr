import os
import cv2
from id_extractor import IdExtractor
from id_structs import IdKeypoints
from pprint import pprint
from id_utils import perspective_transform_from_kpts
from argparse import ArgumentParser
import uuid

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, help="target folder")
    # parser.add_argument("--img", type=str, help="DA IMAGE")
    opts = parser.parse_args()
    for image in os.listdir(opts.folder):
        if image.endswith((".jpeg", ".jpg", ".png", ".JPG")):
            img = cv2.imread(os.path.join(opts.folder,image))
            custom_config = r"-l khm --oem 3 --psm 6"
            extractor = IdExtractor("./models/id_kpt_vn.pt", "./models/id_ki.pt")
            # _results = extractor.extract_end2end(img, debug=True)
            kpts = extractor.get_keypoints(img)
            try:
                tranform_img = perspective_transform_from_kpts(kpts, img)
                print(tranform_img.shape)
                cv2.imwrite(f"debug_crops/transform_{uuid.uuid4().hex}.png", tranform_img)
            except Exception as _e:
                continue
            # pprint(_results)
