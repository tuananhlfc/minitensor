import unittest
import numpy as np
from parameterized import parameterized
from minitensor.tensor_data import TensorData, IndexingError, strides_from_shape


class TestTensorData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up shared resources for all tests
        cls.storage = [1.0, 2.0, 3.0, 4.0]
        cls.shape = (2, 2)
        cls.strides = strides_from_shape(cls.shape)

    def setUp(self):
        # Set up a fresh TensorData instance for each test
        self.tensor = TensorData(self.storage, self.shape)

    def test_initialization(self):
        """Test initialization of TensorData."""
        self.assertEqual(self.tensor.shape, self.shape)
        self.assertEqual(self.tensor.strides, self.strides)
        self.assertEqual(self.tensor.size, 4)
        self.assertTrue(
            np.array_equal(
                self.tensor._storage, np.array(self.storage, dtype=np.float32)
            )
        )

    @parameterized.expand(
        [
            ((0, 0), 0),
            ((0, 1), 1),
            ((1, 0), 2),
            ((1, 1), 3),
        ]
    )
    def test_index_to_position(self, index, expected_position):
        """Test index-to-position conversion."""
        position = self.tensor.index(index)
        self.assertEqual(position, expected_position)

    def test_get_and_set(self):
        """Test getting and setting values."""
        self.tensor.set((0, 1), 10.0)
        value = self.tensor.get((0, 1))
        self.assertEqual(value, 10.0)

    def test_is_contiguous(self):
        """Test if the tensor is contiguous."""
        self.assertTrue(self.tensor.is_contiguous())

    def test_indices(self):
        """Test generating all indices."""
        indices = list(self.tensor.indices())
        expected_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.assertEqual(indices, expected_indices)

    def test_sample(self):
        """Test sampling a random index."""
        sample_index = self.tensor.sample()
        self.assertTrue(
            all(0 <= idx < dim for idx, dim in zip(sample_index, self.shape))
        )

    def test_permute(self):
        """Test permuting dimensions."""
        permuted_tensor = self.tensor.permute(1, 0)
        self.assertEqual(permuted_tensor.shape, (2, 2))
        self.assertEqual(permuted_tensor.strides, (1, 2))

    @parameterized.expand(
        [
            ((2, 1), (1, 3), (2, 3)),
            ((3, 1, 4), (1, 5, 4), (3, 5, 4)),
        ]
    )
    def test_shape_broadcast(self, shape1, shape2, expected_shape):
        """Test shape broadcasting."""
        broadcasted_shape = TensorData.shape_broadcast(shape1, shape2)
        self.assertEqual(broadcasted_shape, expected_shape)

    def test_indexing_error(self):
        """Test invalid indexing."""
        with self.assertRaises(IndexingError):
            self.tensor.index((2, 2))

    def test_to_string(self):
        """Test string representation of the tensor."""
        tensor_str = self.tensor.to_string()
        self.assertIn("[1.000000 2.000000]", tensor_str)
        self.assertIn("[3.000000 4.000000]", tensor_str)


if __name__ == "__main__":
    unittest.main()
