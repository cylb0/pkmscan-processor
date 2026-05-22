import pytest
import numpy as np
from pokemon_card_processor import *
from unittest.mock import MagicMock, patch
from errors.processing import PokemonCardDetectionError

class TestPokemonCardProcessor:

    @pytest.fixture
    @patch("pokemon_card_processor.YOLO")
    def processor(self, mock_yolo):
        return PokemonCardProcessor("dummy.pt")

    class TestInit:
        @patch("pokemon_card_processor.YOLO")
        def test_sets_correct_dimensions(self, mock_yolo):
            processor = PokemonCardProcessor("dummy_model.pt")

            assert processor.aspect_ratio == 63 / 88
            assert processor.target_height == 700
            assert processor.target_width == 501
            assert processor.axis_treshold == 0.05
            assert processor.min_area == pytest.approx(700 * 501 * 0.3)

    class TestCheckResolution:
        def test__nominal(self, processor):
            quad = np.array(
                [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
                dtype=np.int32
            )
            processor._check_resolution(quad)

        def test_too_low_raises_exception(self, processor):
            quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)

            with pytest.raises(PokemonCardDetectionError):
                processor._check_resolution(quad)

    class TestLoadImage:
        @patch("pokemon_card_processor.cv2.imread")
        def test_nominal(self, mock_imread, processor):
            fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_imread.return_value = fake_img
            
            result = processor._load_image("fake/img.jpg")

            assert isinstance(result, np.ndarray)
            assert result.shape == (100, 100, 3)
            mock_imread.assert_called_once_with("fake/img.jpg")

        @patch("pokemon_card_processor.cv2.imread")
        def test_fail_raises_value_error(self, mock_imread, processor):
            mock_imread.return_value = None

            with pytest.raises(ValueError, match="Failed to read image"):
                processor._load_image("invalid/img.jpg")

    class TestDetectCardMask:
        def test_nominal(self, processor):
            fake_pts = np.array([[10.5, 20.5], [100.2, 20.1], [100.8, 150.3], [10.1, 150.9]], dtype=np.float32)
            mock_result = MagicMock()
            mock_result.boxes = [MagicMock()]
            mock_result.masks.xy = [fake_pts]

            processor.model.return_value = [mock_result]

            fake_img = np.zeros((500, 500, 3), dtype=np.uint8)
            result = processor._detect_card_mask(fake_img)

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.int32
            assert result[0][0] == 10
            assert len(result) == 4
        
        def test_no_detection(self, processor):
            mock_result = MagicMock()
            mock_result.boxes= []
            processor.model.return_value = [mock_result]

            fake_img = np.zeros((500, 500, 3), dtype=np.uint8)

            with pytest.raises(PokemonCardDetectionError, match="No card found"):
                processor._detect_card_mask(fake_img)
        
    class TestRetrieveAndValidatePolygon:
        def test_nominal(self, processor):
            pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)

            result = processor._retrieve_and_validate_polygon(pts)

            assert len(result) == 4
            assert cv2.isContourConvex(result)

        def test_wrong_number_vertices(self, processor):
            pts = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.int32)

            with pytest.raises(PokemonCardDetectionError, match="Image rejected, found 3 vertices"):
                processor._retrieve_and_validate_polygon(pts)

        def test_simplification_logic(self, processor):
            pts = np.array([
                [0, 0], [20, 0], [80, 0],
                [100, 0], [100, 40], [100, 80],
                [100, 100], [70, 100], [20, 100], [10, 100],
                [0, 100], [0, 50]
            ], dtype=np.int32)

            result = processor._retrieve_and_validate_polygon(pts)
            assert len(result) == 4

        def test_non_convex(self, processor):
            pts = np.array([[0, 0], [100, 0], [0, 100], [30, 30]], dtype=np.int32)

            with pytest.raises(PokemonCardDetectionError, match="not convex"):
                processor._retrieve_and_validate_polygon(pts)

    class TestStraightenCard:
        @patch("pokemon_card_processor.get_perspective_matrix")
        @patch("pokemon_card_processor.order_quad_points")
        def test_nominal(self, mock_order, mock_matrix, processor):
            mock_order.return_value = np.zeros((4, 2), dtype=np.float32)
            mock_matrix.return_value = np.eye(3, dtype=np.float32)

            fake_img = np.zeros((500, 500, 3), dtype=np.uint8)
            quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)

            result = processor._straighten_card(fake_img, quad)

            assert result.shape == (700, 501, 3)
            assert result.dtype == np.uint8

        @patch("pokemon_card_processor.cv2.warpPerspective")
        def test_uses_correct_lanczos4_interpolation(self, mock_warp, processor):
            fake_img = np.zeros((500, 500, 3), dtype=np.uint8)
            quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)

            processor._straighten_card(fake_img, quad)
            
            args, kwargs = mock_warp.call_args
            assert kwargs["flags"] == cv2.INTER_LANCZOS4
    