import numpy as np
from id_structs import IdKeypoints
import cv2
import os


def create_id_keypoints(pos, coords):
    return {"position": pos, "coordinates": coords}


def draw_rectangle(image, top_left, bottom_right, color=(255, 0, 0), thickness=2):
    """does what it says (draw rect)"""
    return cv2.rectangle(image, top_left, bottom_right, color, thickness)


def get_rect_coords(input_coords: list):
    """
    takes in a list of x1y1 (top left) and x2y2 (bottom right)
    returns `top_left, top_right, bottom_left, bottom_right`
    """
    top_left = input_coords[0]
    bottom_right = input_coords[1]
    top_right = [bottom_right[0], top_left[1]]  # (x2 y1)
    bottom_left = [top_left[0], bottom_right[1]]  # (x1 y2)
    return top_left, top_right, bottom_left, bottom_right


def calculate_center(input_coords):
    """
    calculates the center between two points
    takes in a list of x1y1  and x2y2
    returns center
    """
    top_left = input_coords[0]
    bottom_right = input_coords[1]
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Calculate center coordinates
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return (cx, cy)

def perspective_transform_from_kpts_nocrop(
    extracted_keypoints: IdKeypoints, original_image: np.ndarray
):
    keypoint_dict = {}
    keypoint_dict["TOP_LEFT"] = extracted_keypoints.top_left
    keypoint_dict["TOP_RIGHT"] = extracted_keypoints.top_right
    keypoint_dict["BOTTOM_LEFT"] = extracted_keypoints.bottom_left
    keypoint_dict["BOTTOM_RIGHT"] = extracted_keypoints.bottom_right

    # Get source points in the order: top-left, top-right, bottom-right, bottom-left
    src_points = np.float32(
        [
            keypoint_dict["TOP_LEFT"],
            keypoint_dict["TOP_RIGHT"],
            keypoint_dict["BOTTOM_RIGHT"],
            keypoint_dict["BOTTOM_LEFT"],
        ]
    )

    # Define destination points covering the full original image
    height, width = original_image.shape[:2]
    dst_points = np.float32(
        [
            [0, 0],  # top-left
            [width - 1, 0],  # top-right
            [width - 1, height - 1],  # bottom-right
            [0, height - 1],  # bottom-left
        ]
    )

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(
        original_image, 
        matrix, 
        (width, height), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0)
    )
    return result
def perspective_transform_from_kpts(
    extracted_keypoints: IdKeypoints, original_image: np.ndarray
):
    keypoint_dict = {}
    keypoint_dict["TOP_LEFT"] = extracted_keypoints.top_left
    keypoint_dict["TOP_RIGHT"] = extracted_keypoints.top_right
    keypoint_dict["BOTTOM_LEFT"] = extracted_keypoints.bottom_left
    keypoint_dict["BOTTOM_RIGHT"] = extracted_keypoints.bottom_right

    # Get source points in the order: top-left, top-right, bottom-right, bottom-left
    src_points = np.float32(
        [
            keypoint_dict["TOP_LEFT"],
            keypoint_dict["TOP_RIGHT"],
            keypoint_dict["BOTTOM_RIGHT"],
            keypoint_dict["BOTTOM_LEFT"],
        ]
    )

    # Calculate width and height for the destination image
    # Using the maximum distance between horizontal and vertical points
    width = max(
        np.linalg.norm(
            np.array(keypoint_dict["TOP_RIGHT"]) - np.array(keypoint_dict["TOP_LEFT"])
        ),
        np.linalg.norm(
            np.array(keypoint_dict["BOTTOM_RIGHT"])
            - np.array(keypoint_dict["BOTTOM_LEFT"])
        ),
    )

    height = max(
        np.linalg.norm(
            np.array(keypoint_dict["BOTTOM_LEFT"]) - np.array(keypoint_dict["TOP_LEFT"])
        ),
        np.linalg.norm(
            np.array(keypoint_dict["BOTTOM_RIGHT"])
            - np.array(keypoint_dict["TOP_RIGHT"])
        ),
    )

    # Define destination points (rectangle)
    dst_points = np.float32(
        [
            [0, 0],  # top-left
            [width - 1, 0],  # top-right
            [width - 1, height - 1],  # bottom-right
            [0, height - 1],  # bottom-left
        ]
    )

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(original_image, matrix, (int(width), int(height)))

    return result


def calculate_rotation_angle(p1, p2):
    """
    get rotation angle between two points.
    takes in two points returns a float for angle
    """
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    return np.arctan2(delta_y, delta_x) * 180.0 / np.pi
