from ultralytics import YOLO
import uuid
from id_utils import calculate_center, perspective_transform_from_kpts
import cv2
import os
from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
from id_structs import IdKeyInformation, IdKeypoints, IdInfo
import pytesseract


class IdExtractor:
    """
    todo: refactor the get functions
    """

    def __init__(self, kpt_model_path: str, ki_model_path: str):
        self.kpt_model = YOLO(kpt_model_path)
        self.ki_model = YOLO(ki_model_path)
        self.warmup()

    def warmup(self):
        """
        run a forward pass on a dummy image, so that *IF* the model is loaded on to the GPU (at least CUDA enabled ones) , it will be fast on the actual usage!
        """
        test_img = np.zeros([640, 640, 3])
        self.kpt_model(test_img, verbose=False)
        self.ki_model(test_img, verbose=False)

    def get_keyinfo(self, input_image: np.ndarray, conf_thresh: float = 0.6) -> IdKeyInformation:
        key_info = IdKeyInformation(
            address=None,
            appearance=None,
            birth=None,
            birthplace=None,
            bottom=None,
            expiry=None,
            gender=None,
            height=None,
            id_num=None,
            name=None,
            name_latin=None,
        )
        extracted_key_info = self.ki_model(input_image, conf=conf_thresh)[0]
        for result in extracted_key_info:
            classes = result.boxes.cls.tolist()
            xyxys = result.boxes.xyxy.tolist()
            for idx, xyxy in enumerate(xyxys):
                class_name = self.ki_model.names[classes[idx]]
                x1, y1, x2, y2 = [x for x in xyxy]
                ki_pos_ = (x1, y1, x2, y2)
                if class_name == "address":
                    key_info.address = ki_pos_
                if class_name == "appearance":
                    key_info.appearance = ki_pos_
                if class_name == "birth":
                    key_info.birth = ki_pos_
                if class_name == "birthplace":
                    key_info.birthplace = ki_pos_
                if class_name == "bottom":
                    key_info.bottom = ki_pos_
                if class_name == "expiry":
                    key_info.expiry = ki_pos_
                if class_name == "gender":
                    key_info.gender = ki_pos_
                if class_name == "height":
                    key_info.height = ki_pos_
                if class_name == "id_num":
                    key_info.id_num = ki_pos_
                if class_name == "name":
                    key_info.name = ki_pos_
                if class_name == "name_latin":
                    key_info.name_latin = ki_pos_
        return key_info

    def get_keypoints(self, input_image: np.ndarray, conf_thresh: float = 0.6) -> IdKeypoints:
        """
        get keypoints from image
        """
        kpts = IdKeypoints(
            top_left=None, top_right=None, bottom_left=None, bottom_right=None
        )
        extracted_keypoints = self.kpt_model(input_image, conf=conf_thresh)[0]
        for result in extracted_keypoints:
            classes = result.boxes.cls.tolist()
            xyxys = result.boxes.xyxy.tolist()
            for idx, xyxy in enumerate(xyxys):
                class_name = self.kpt_model.names[classes[idx]]
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                kpt_center_ = calculate_center([[x1, y1], [x2, y2]])
                if class_name == "TOP_LEFT":
                    kpts.top_left = kpt_center_
                if class_name == "TOP_RIGHT":
                    kpts.top_right = kpt_center_
                if class_name == "BOTTOM_LEFT":
                    kpts.bottom_left = kpt_center_
                if class_name == "BOTTOM_RIGHT":
                    kpts.bottom_right = kpt_center_
        return kpts

    def extract_end2end(
        self, input_image: np.ndarray, 
        backend: str = "TESSERACT", 
        debug: bool = False
    ) -> dict:
        if debug:
            os.makedirs("debug_crops", exist_ok=True)
        info = {}
        # todo handle diagonal only points with `fill_missing_inferred_rect`
        keypoints = self.get_keypoints(input_image)
        print(keypoints)
        kpts_good = keypoints.check_if_filled()
        if kpts_good:
            translated_info = perspective_transform_from_kpts(keypoints, input_image)
        else:
            translated_info = input_image 
        cv2.imwrite("debug_crops/translated.png", translated_info)
        keyinfo = self.get_keyinfo(translated_info, 0.3)
        pprint(keyinfo)

        # tesseract
        # custom_config = r"-l khm-dset --oem 3 --psm 6"
        custom_config = r"-l khm --psm 6"
        # custom_config = r"-l vie --oem 3 --psm 6"
        en_config = r"-l eng --oem 3 --psm 6"

        for _a in keyinfo:
            field_name, coords = _a
            if coords is not None:
                x1, y1, x2, y2 = [int(x) for x in coords]
                crop_img = translated_info[y1:y2, x1:x2]
                if debug:
                    cv2.imwrite(f"debug_crops/{field_name}_{uuid.uuid4().hex}.png", crop_img)
                if field_name == "name_latin" or field_name == "id_num":
                    result = pytesseract.image_to_string(
                        crop_img, config=en_config
                    ).split("\n")
                else:
                    result = pytesseract.image_to_string(
                        crop_img, config=custom_config
                    ).split("\n")
                info[field_name] = result
        return info
