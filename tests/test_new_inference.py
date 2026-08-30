"""Guard the new experiments' boundaries and the upload highlight endpoint."""
import unittest
from foodvision.data_budget import eligible, expanded_episode

class ExperimentChecks(unittest.TestCase):

    def test_expansion_purges_cross_role_images_and_group_links(self):
        rows = [{'record_id': str(i), 'source': 's', 'group': str(i), 'similarity_component': str(i)} for i in range(12)]
        rows[1]['group'] = '0'
        rows[2]['similarity_component'] = '0'
        hashes = {'0': {'before', 'after'}, '3': {'other', 'before'}}
        self.assertEqual(eligible(rows, list(range(12)), [0], hashes), list(range(4, 12)))
        episode = expanded_episode(rows, {'source': 's', 'name': 'test', 'test': [0]}, hashes)
        self.assertEqual(episode['train'], list(range(4, 12)))
        self.assertCountEqual([i for f in episode['inner'] for i in f['validation']], episode['train'])
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
