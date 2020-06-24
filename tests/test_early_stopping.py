import os
import torchtestcase
from mlearn.utils.early_stopping import EarlyStopping
from mlearn.modeling.embedding import RNNClassifier


class TestEarlyStopping(torchtestcase.TorchTestCase):
    """Test Early stopping module."""

    @classmethod
    def setUp(cls):
        """Set up Early Stopping test class."""
        cls.scores_high = [0.2, 0.1, 0.4, 0.4, 0.2, 0.1, 0.3, 0.32]
        cls.scores_low = [0.2, 0.1, 0.4, 0.4, 0.2, 0.11, 0.3, 0.32]
        cls.path_prefix = 'tests/earlystop'
        cls.model = RNNClassifier(20, 20, 20, 20, 0.1, True)
        cls.initialised = {'patience': None, 'best_model': None, 'best_score': None, 'best_epoch': 0, 'epoch': 0,
                           'low_is_good': None, 'path_prefix': f'{cls.path_prefix}',
                           'verbose': False, 'model': cls.model}

    @classmethod
    def tearDown(cls):
        """Tear down class between each test."""
        cls.initialised = {'patience': None, 'best_model': None, 'best_score': None, 'best_epoch': 0, 'epoch': 0,
                           'low_is_good': None, 'path_prefix': f'{cls.path_prefix}',
                           'verbose': False, 'model': cls.model}
        if os.path.exists('tests/earlystop_rnn.mdl'):
            os.remove('tests/earlystop_rnn.mdl')

    def test_initalize(self):
        """Test EarlyStopping.__init__()."""
        early_stop_low = EarlyStopping(self.path_prefix, self.model, patience = 5, low_is_good = True)
        self.initialised.update({'patience': 5, 'low_is_good': True})

        self.assertIsInstance(early_stop_low, EarlyStopping, msg = "EarlyStopping not initiated correctly.")
        self.assertDictEqual(early_stop_low.__dict__, self.initialised, msg = "A variable is initialized incorrectly.")

    def test_set_best(self):
        """Test EarlyStopping.best_state."""
        early_stop = EarlyStopping(self.path_prefix, self.model, patience = 5, low_is_good = True)
        early_stop.best_state = self.model
        self.assertTrue(os.path.exists(f'{early_stop.path_prefix}_{self.model.name}.mdl'),
                        msg = "Model is not saved properly.")

        model = early_stop.best_state
        self.assertEqual(self.model, model, msg = "Loaded model does not equal stored model.")

    def test_new_best(self):
        """Test EarlyStopping.new_best()."""
        early_stop_low = EarlyStopping(self.path_prefix, self.model, patience = 5, low_is_good = True)
        early_stop_low.best_score = 0.5
        self.assertTrue(early_stop_low.new_best(0.2), msg = "Lower score not set as new best for (EarlyStopping low).")
        self.assertTrue(early_stop_low.new_best(0.5), msg = "Same score not set as new best (EarlyStopping low).")
        self.assertFalse(early_stop_low.new_best(0.8), msg = "Higher value set as being better (EarlyStopping low).")

        early_stop_high = EarlyStopping(self.path_prefix, self.model, patience = 5, low_is_good = False)
        early_stop_high.best_score = 0.5
        self.assertTrue(early_stop_high.new_best(0.8), msg = "Higher score not set as new best (EarlyStopping high).")
        self.assertTrue(early_stop_high.new_best(0.5), msg = "Same score not set as new best for (EarlyStopping high).")
        self.assertFalse(early_stop_high.new_best(0.3), msg = "Lower value is set as being better.")

    def test_call(self):
        """Test EarlyStopping.__call__()"""
        early_stop_low = EarlyStopping(self.path_prefix, self.model, patience = 5, low_is_good = True, verbose = True)
        for score in self.scores_low:
            early_stop_low(self.model, score)
        self.assertEqual(early_stop_low.best_epoch, 2, msg = "EarlyStopping (low) identifies wrong epoch as best.")

        early_stop_high = EarlyStopping(self.path_prefix, self.model, patience = 5, low_is_good = False)
        for score in self.scores_high:
            early_stop_high(self.model, score)
        self.assertEqual(early_stop_high.best_epoch, 4, msg = "EarlyStopping (high) identifies wrong epoch as best.")
