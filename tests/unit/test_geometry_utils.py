import pytest
import cv2
import numpy as np
from utils.geometry_utils import order_quad_points, get_perspective_matrix


class TestOrderQuadPoints:
    def test_nominal_square(self):
        """Should order [0,0], [10,0], [10,10], [0,10] correctly"""
        pts = np.array([[10, 10], [0, 0], [10, 0], [0, 10]], dtype="float32")
        ordered = order_quad_points(pts)

        expected = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        np.testing.assert_array_equal(ordered, expected)

    def test_rotated_nominal_square(self):
        pts = np.array([[0, 10], [10, 10], [0, 0], [10, 0]], dtype="float32")
        ordered = order_quad_points(pts)

        expected = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        np.testing.assert_array_equal(ordered, expected)
        pass

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            order_quad_points(np.array([[0, 10], [10, 10], [0, 0]], dtype="float32"))

    def test_centroid_is_origin(self):
        pts = np.array([[10, 10], [-10, -10], [10, -10], [-10, 10]], dtype="float32")
        ordered = order_quad_points(pts)

        assert np.array_equal(ordered[0], [-10, -10])
        assert np.array_equal(ordered[2], [10, 10])

    def test_output_float32(self):
        pts = np.array([[10, 10], [-10, -10], [10, -10], [-10, 10]], dtype="int32")
        ordered = order_quad_points(pts)
        assert ordered.dtype == np.float32

    def test_negative_coords(self):
        pts = np.array([[-20, -10], [-10, -10], [-10, -20], [-20, -20]], dtype="int32")
        ordered = order_quad_points(pts)

        expected = np.array(
            [[-20, -20], [-10, -20], [-10, -10], [-20, -10]], dtype="float32"
        )
        np.testing.assert_array_equal(ordered, expected)


class TestGetPerspectiveMatrix:
    def test_matrix_shape_and_type(self):
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")
        matrix = get_perspective_matrix(pts, 100, 100)

        assert matrix.shape == (3, 3)
        assert matrix.dtype == np.float64

    def test_mapping_accuracy(self):
        width, height = 630, 880
        src_pts = np.array(
            [[10, 10], [600, 20], [620, 850], [50, 879]], dtype="float32"
        )
        matrix = get_perspective_matrix(src_pts, width, height)

        test_pt = np.array(
            [src_pts[0][0], src_pts[0][1], 1.0]
        )  # Homogeneous coordinates (x, y, 1)
        res = matrix @ test_pt
        res /= res[2]  # Normalization to (x, y) coordinates

        np.testing.assert_array_almost_equal(res[:2], [0, 0], decimal=2)

    def test_br_mapping(self):
        width, height = 630, 880
        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")

        matrix = get_perspective_matrix(src_pts, width, height)

        test_pts = src_pts.reshape(-1, 1, 2)
        dst_pts = cv2.perspectiveTransform(test_pts, matrix).reshape(-1, 2)

        expected = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )

        np.testing.assert_array_almost_equal(dst_pts, expected, decimal=2)
