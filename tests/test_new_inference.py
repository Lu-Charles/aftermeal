"""Guard the new experiments' boundaries and the upload highlight endpoint."""
import io
import unittest
from PIL import Image
from foodvision.inference import decode_image

class PhotoOrientationChecks(unittest.TestCase):

    def test_exif_orientation_is_applied(self):
        stream = io.BytesIO()
        photo = Image.new('RGB', (230, 250))
        exif = photo.getexif()
        exif[274] = 6
        photo.save(stream, format='JPEG', exif=exif)
        self.assertEqual(decode_image(stream.getvalue()).size, (250, 230))
