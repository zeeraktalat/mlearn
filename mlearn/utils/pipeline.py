from mlearn import base
from mlearn.data_processing.data import GeneralDataset
from mlearn.data_processing.batching import Batch, BatchExtractor


def process_and_batch(dataset: GeneralDataset, data: base.DataType, batch_size: int, onehot: bool = True):
    """
    Process a dataset and data.

    :dataset (GeneralDataset): The dataset object to use for processing.
    :data (base.DataType): The data to be batched and processed.
    :batch_size (int): Size of batches to create.
    :returns: Batched data.
    """
    # Process labels and encode data.
    dataset.process_labels(data)

    # Batch data
    batch = Batch(batch_size, data)
    batch.create_batches()
    batches = BatchExtractor('label', batch, dataset, onehot)
    return batches
