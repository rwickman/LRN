
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tf.to_float = lambda x: tf.cast(x, tf.float32)

def get_all_datapoints(dataset):
  """Returns all datapoints in a dataset.

  Args:
    dataset: (tf.data.Dataset) dataset containing the images to hash.

  Returns:
    A list of datapoints returned from the dataset.
  """
  session = tf.Session(graph=tf.get_default_graph())
  iterator = dataset.make_one_shot_iterator().get_next()

  data = []

  try:
    while True:
      data.append(session.run(iterator))
  except tf.errors.OutOfRangeError:
    pass

  return data


def make_negative(data):
  data['image'] = 1. - data['image']
  return data

def normalize(data):
  data['image'] = tf.to_float(data['image']) / 255.
  return data

def convert_to_grayscale(data):
  data['image'] = tf.image.rgb_to_grayscale(data['image'])
  return data

def format_omniglot(data):
  """Formats an Omniglot input into a standard format.

  The formatted sample will have two keys: 'image', containing the input image,
  and 'label' containing the label.

  Args:
    data: dict, a sample from the Omniglot dataset. Contains keys: 'image',
      'alphabet' and 'alphabet_char_id'.

  Returns:
    Formatted `data` dict.
  """
  data['label'] = data['alphabet_char_id']

  del data['alphabet_char_id']
  del data['alphabet']

  return data

def get_omniglot_order():
  """Returns the omniglot alphabet names, in the order used in previous works.

  Returns:
    Alphabet names of the 50 Omniglot tasks, in the same order as used by
    multiple previous works, such as "Diversity and Depth in Per-Example
    Routing Models" (https://openreview.net/pdf?id=BkxWJnC9tX).
  """
  return [
      'Gujarati', 'Sylheti', 'Arcadian', 'Tibetan',
      'Old_Church_Slavonic_(Cyrillic)', 'Angelic', 'Malay_(Jawi_-_Arabic)',
      'Sanskrit', 'Cyrillic', 'Anglo-Saxon_Futhorc', 'Syriac_(Estrangelo)',
      'Ge_ez', 'Japanese_(katakana)', 'Keble', 'Manipuri',
      'Alphabet_of_the_Magi', 'Gurmukhi', 'Korean', 'Early_Aramaic',
      'Atemayar_Qelisayer', 'Tagalog', 'Mkhedruli_(Georgian)',
      'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Tengwar', 'Hebrew', 'N_Ko',
      'Grantha', 'Latin', 'Syriac_(Serto)', 'Tifinagh', 'Balinese', 'Mongolian',
      'ULOG', 'Futurama', 'Malayalam', 'Oriya',
      'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Avesta', 'Kannada', 'Bengali',
      'Japanese_(hiragana)', 'Armenian', 'Aurek-Besh', 'Glagolitic',
      'Asomtavruli_(Georgian)', 'Greek', 'Braille', 'Burmese_(Myanmar)',
      'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Atlantean'
  ]


def map_omniglot_alphabet_names_to_ids(alphabet_names):
  """Maps a list of Omniglot alphabet names into their ids.

  Args:
    alphabet_names: (list of strings) names of Omniglot alphabets to be mapped.

  Returns:
    A list of ints, corresponding to alphabet ids for the given names. All ids
    are between 0 and 49 inclusive.
  """
  _, info = tfds.load(name='omniglot', split=tfds.Split.ALL, with_info=True)

  alphabet_ids = [
      info.features['alphabet'].str2int(alphabet_name)
      for alphabet_name in alphabet_names
  ]

  return alphabet_ids


def get_omniglot(alphabet_id, size=None):
  """Loads one of the Omniglot alphabets.

  Args:
    alphabet_id: (int) alphabet id, must be between 0 and 49 inclusive.
    size: either None or a pair of ints. If set to None, the images will not
      be resized, and will retain their original size of 105x105. If set to
      a pair of ints, then the images will be resized to this size.

  Returns:
    A pair of:
      - a dictionary with keys: 'train', 'validation' and 'test'. Values for
        these keys are `tf.data.Dataset`s for train, validation and test,
        respectively.
      - an integer denoting the number of classes in the dataset
  """
  assert 0 <= alphabet_id < 50
  np.random.seed(seed=alphabet_id)

  pred = lambda datapoint: tf.math.equal(datapoint['alphabet'], alphabet_id)

  # The `as_dataset_kwargs` argument makes this function deterministic.
  dataset = tfds.load(
      name='omniglot',
      split=tfds.Split.ALL,
      as_dataset_kwargs={'shuffle_files': False})

  dataset = dataset.filter(pred)
  dataset = dataset.map(format_omniglot)
  dataset = dataset.map(normalize)
  dataset = dataset.map(convert_to_grayscale)

  # Flip to make the background consist of 0's and characters consist of 1's
  # (instead of the other way around).
  dataset = dataset.map(make_negative)

  if size:
    dataset = dataset.map(resize(size))

  all_datapoints = get_all_datapoints(dataset)
  num_classes = max([datapoint['label'] for datapoint in all_datapoints]) + 1

  data = {data_split: [] for data_split in _ALL_SPLITS}

  for data_group in group_by_label(all_datapoints, num_classes):
    group_size = len(data_group)

    # Make sure that the datapoints can be divided evenly.
    assert group_size % 10 == 0

    np.random.shuffle(data_group)

    train_size = int(0.5 * group_size)
    validation_size = int(0.2 * group_size)

    data['train'] += data_group[:train_size]
    data['validation'] += data_group[train_size:train_size+validation_size]
    data['test'] += data_group[train_size+validation_size:]

  train_size = len(data['train'])

  for split in _ALL_SPLITS:
    data[split] = convert_list_to_dataset(data[split])

  # Ensure that the order of training data is different in every epoch.
  data['train'] = data['train'].shuffle(buffer_size=train_size)

  return data, num_classes

def get_data_for_multitask_omniglot_setup(num_alphabets):
  """Loads a given number of Omniglot datasets for multitask learning.

  Args:
    num_alphabets: (int) number of alphabets to use, must be between 1 and 50
      inclusive.

  Returns:
    A pair of two lists (`task_data`, `num_classes_for_tasks`), each
    containing one element per alphabet. These lists respectively contain
    task input data and number of classification classes, as returned from
    `get_omniglot`.
  """
  alphabet_names = get_omniglot_order()[:num_alphabets]
  alphabet_ids = map_omniglot_alphabet_names_to_ids(alphabet_names)

  alphabets = [get_omniglot(alphabet_id) for alphabet_id in alphabet_ids]

  # Convert a list of pairs into a pair of lists and return
  return [list(tup) for tup in zip(*alphabets)]


print(get_data_for_multitask_omniglot_setup(20))