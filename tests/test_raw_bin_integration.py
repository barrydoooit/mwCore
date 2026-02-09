
import sys
import os
import unittest
import numpy as np

# Ensure mwCore is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mwcore.registry import READERS
from mwcore.radario.readers.offlineReaders.raw_bin_reader import RawBinReader

class TestRawBinIntegration(unittest.TestCase):
    def test_registry_instantiation(self):
        """Test that we can build the reader from the registry."""
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'sample.bin')
        
        cfg = dict(
            type='RawBinReader',
            file_path=file_path
        )
        
        reader = READERS.build(cfg)
        self.assertIsInstance(reader, RawBinReader)
        return reader

    def test_read_frame(self):
        """Test reading and processing a frame."""
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'sample.bin')
        reader = RawBinReader(file_path=file_path)
        
        # Read first frame
        points = reader.read()
        
        self.assertIsNotNone(points)
        self.assertTrue(isinstance(points, np.ndarray))
        # Expect (6, N) shape
        self.assertEqual(points.shape[0], 6)
        
        print(f"Successfully read frame with {points.shape[1]} points.")
        reader.close()

if __name__ == '__main__':
    unittest.main()
