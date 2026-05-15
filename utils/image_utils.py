import numpy as np
import cv2


def save_as_webp(img: np.ndarray, output_path: str, quality: int = 80) -> None:
    success = cv2.imwrite(output_path, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not success:
        raise RuntimeError(f"Failed to save WebP image to {output_path}")
