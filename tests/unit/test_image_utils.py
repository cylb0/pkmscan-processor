import pytest
import numpy as np
from utils.image_utils import save_as_webp


class TestSaveAsWebP:
    def test_save_as_webp_creates_file(self, tmp_path):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        output = tmp_path / "test.webp"

        save_as_webp(img, str(output))
        assert output.exists()
        assert output.stat().st_size > 0

    def test_save_as_webp_raises_error_on_invalid_path(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="Failed to save WebP"):
            save_as_webp(img, "/non_existing_path/res.webp")
