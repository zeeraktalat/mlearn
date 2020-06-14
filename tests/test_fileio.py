import unittest
from mlearn.data import fileio as io


class TestFileIO(unittest.TestCase):
    """Test FileIO.py."""

    def test_read_json(self):
        """Test fileio.read_json()"""
        filepath = 'tests/data/traindeep.json'
        doc_key, label_key = 'text', 'label'
        expected = [{'text': 'me gusta comer en la cafeteria', 'label': 'spanish', 'username': 'zeerakw',
                     'name': 'zeerak', 'place': 'some place'},
                    {'text': 'give it to me', 'label': 'english', 'username': 'madeup',
                     'name': 'made', 'place': 'far far away'},
                    {'text': 'no creo que sea una buena idea', 'label': 'spanish', 'username': 'upmade',
                     'name': 'up', 'place': 'long lost'},
                    {'text': 'no it is not a good idea to get lost at sea', 'label': 'english', 'username': 'notreal',
                     'name': 'unreal', 'place': 'rock and hard place'}
                    ]
        result = io.read_json(filepath, 'utf-8', doc_key, label_key,
                              secondary_keys = {'username': 'user|username',
                                                'name': 'user|name',
                                                'place': 'user|location|place'})
        for res, exp in zip(result, expected):
            self.assertDictEqual(res, exp, msg = "A dictionary is not loaded correctly.")

    def test_write_results(self):
        """Test fileio.write_results()"""
        pass

    def test_write_predictions(self):
        """Test fileio.write_predictions()"""
        pass
