from ultralytics import YOLO
import numpy as np
import cv2
from errors.processing import PokemonCardDetectionError
from utils.geometry_utils import get_perspective_matrix, order_quad_points


class PokemonCardProcessor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.aspect_ratio = 63 / 88
        self.axis_treshold = 0.05
        self.target_height = 700
        self.target_width = int(self.target_height * self.aspect_ratio)
        self.min_area = self.target_height * self.target_width * 0.3

    def process(self, input_path: str) -> np.ndarray:
        img = self._load_image(input_path)
        mask_points = self._detect_card_mask(img)
        quad = self._retrieve_and_validate_polygon(mask_points)
        self._check_resolution(quad)

        return self._straighten_card(img, quad.reshape(4, 2))

    def _check_resolution(self, quad: np.ndarray):
        area = cv2.contourArea(quad)
        if area < self.min_area:
            raise PokemonCardDetectionError(
                f"Source resolution is too low ({int(area)}px)."
                "Need at least {self.min_area}px for a quality asset."
            )

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image at {path}")
        return img

    def _detect_card_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Uses vision model to try and extract a mask to be returned as an array of points.
        """
        results = self.model(img)

        if not results or len(results[0].boxes) == 0:
            raise PokemonCardDetectionError("No card found")

        return results[0].masks.xy[0].astype(np.int32)

    def _retrieve_and_validate_polygon(self, points: np.ndarray):
        """
        Simplifies points array into a approximated polygon.
        Makes sure the polygon is a convex 4 points quadrilateral.
        """
        epsilon = 0.02 * cv2.arcLength(points, True)
        approx = cv2.approxPolyDP(points, epsilon, True)

        if len(approx) != 4:
            raise PokemonCardDetectionError(
                f"Image rejected, found {len(approx)} vertices instead of 4 "
                "Please make sure all 4 corners are visible with no obstruction."
            )

        if not cv2.isContourConvex(approx):
            raise PokemonCardDetectionError(
                "Image rejected, the detected polygon is not convex."
                "Please make sure the card is not folded or there is no object overlapping the card."
            )

        return approx

    def _straighten_card(self, img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """
        Applies perspective transformation to reach normalized target size.
        """
        pts = order_quad_points(quad.reshape(4, 2).astype(np.float32))

        m = get_perspective_matrix(pts, self.target_width, self.target_height)
        return cv2.warpPerspective(
            img, m, (self.target_width, self.target_height), flags=cv2.INTER_LANCZOS4
        )
