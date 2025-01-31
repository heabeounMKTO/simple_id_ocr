from typing import Tuple
from pydantic import BaseModel


class IdKeyInformation(BaseModel):
    """
    x, y coordinate pairs for key information
    """

    address: Tuple[float, float, float, float] | None
    appearance: Tuple[float, float, float, float] | None
    birth: Tuple[float, float, float, float] | None
    birthplace: Tuple[float, float, float, float] | None
    bottom: Tuple[float, float, float, float] | None
    expiry: Tuple[float, float, float, float] | None
    gender: Tuple[float, float, float, float] | None
    height: Tuple[float, float, float, float] | None
    id_num: Tuple[float, float, float, float] | None
    name: Tuple[float, float, float, float] | None
    name_latin: Tuple[float, float, float, float] | None


class IdInfo(BaseModel):
    address: str = ""
    appearance: str = ""
    birth: str = ""
    birthplace: str = ""
    bottom: str = ""
    expiry: str = ""
    gender: str = ""
    height: str = ""
    id_num: str = ""
    name: str = ""
    name_latin: str = ""


class IdKeypoints(BaseModel):
    """
    x, y coordinate pairs for keypoints
    """

    top_left: Tuple[float, float] | None
    top_right: Tuple[float, float] | None
    bottom_left: Tuple[float, float] | None
    bottom_right: Tuple[float, float] | None
    
    def check_if_filled(self):
        available_coords = [
            coord for coord in [self.top_left, self.top_right, 
                                self.bottom_left, self.bottom_right] if coord is not None
        ]
        if len(available_coords) != 4:
            return False 
        else:
            return True

    def fill_missing_inferred_rect(self):
        """
        if there is diagonal coordinates,
        returns a complete `IdKeypoints` object by "filling in the blanks"
        """
        available_coords = [
            coord for coord in [self.top_left, self.top_right, 
                                self.bottom_left, self.bottom_right] if coord is not None
        ]

        if len(available_coords) < 2:
            raise ValueError("[ERROR] Insufficient Coordinates!")

        if not (self.top_left and self.top_right and 
                self.bottom_left and self.bottom_right):
            if self.top_left and self.bottom_right:
                # Infer top_right and bottom_left
                top_left_x, top_left_y = self.top_left
                bottom_right_x, bottom_right_y = self.bottom_right
                self.top_right = (bottom_right_x, top_left_y)
                self.bottom_left = (top_left_x, bottom_right_y)
            elif self.top_right and self.bottom_left:
                # Infer top_left and bottom_right
                top_right_x, top_right_y = self.top_right
                bottom_left_x, bottom_left_y = self.bottom_left
                self.top_left = (bottom_left_x, top_right_y)
                self.bottom_right = (top_right_x, bottom_left_y)
            else:
                Exception("[ERROR] No Viable Coordinates Given!")
