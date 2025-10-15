import tensorflow as tf
import numpy as np
import functools
import json
import torch

tf_record = "./tfRecords/test.tfrecord" #Path to record goes here

# not sure what this stuff does but it works
# taken from paper
### --------------------------------------------------------------------------------------------------------------------------------------------
# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}


def convert_to_tensor(x, encoded_dtype):
  if len(x) == 1:
    out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
  else:
    out = []
    for el in x:
      out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
  out = tf.convert_to_tensor(np.array(out))
  return out


def parse_serialized_simulation_example(example_proto, metadata):
  """Parses a serialized simulation tf.SequenceExample.

  Args:
    example_proto: A string encoding of the tf.SequenceExample proto.
    metadata: A dict of metadata for the dataset.

  Returns:
    context: A dict, with features that do not vary over the trajectory.
    parsed_features: A dict of tf.Tensors representing the parsed examples
      across time, where axis zero is the time axis.

  """
  if 'context_mean' in metadata:
    feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
  else:
    feature_description = _FEATURE_DESCRIPTION
  context, parsed_features = tf.io.parse_single_sequence_example(
      example_proto,
      context_features=_CONTEXT_FEATURES,
      sequence_features=feature_description)
  for feature_key, item in parsed_features.items():
    convert_fn = functools.partial(
        convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
    parsed_features[feature_key] = tf.py_function(
        convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

  # There is an extra frame at the beginning so we can calculate pos change
  # for all frames used in the paper.
  position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

  # Reshape positions to correct dim:
  parsed_features['position'] = tf.reshape(parsed_features['position'],
                                           position_shape)
  # Set correct shapes of the remaining tensors.
  sequence_length = metadata['sequence_length'] + 1
  if 'context_mean' in metadata:
    context_feat_len = len(metadata['context_mean'])
    parsed_features['step_context'] = tf.reshape(
        parsed_features['step_context'],
        [sequence_length, context_feat_len])
  # Decode particle type explicitly
  context['particle_type'] = tf.py_function(
      functools.partial(convert_fn, encoded_dtype=np.int64),
      inp=[context['particle_type'].values],
      Tout=[tf.int64])
  context['particle_type'] = tf.reshape(context['particle_type'], [-1])
  return context, parsed_features


def _read_metadata(data_path):
    with open(data_path, 'rt') as fp:
        return json.loads(fp.read())

meta_file = "./tfRecords/metadata.json" #path to metadat.json
metadata = _read_metadata(meta_file)

# Create a tf.data.Dataset from the TFRecord.
ds = tf.data.TFRecordDataset(tf_record)
ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))
### --------------------------------------------------------------------------------------------------------------------------------------------
total_array = []
# here is where we actually are usng the data
for context, parsed_features in ds:  # Adjust the number to inspect more or fewer examples
    # print("Context:")
    # for key, value in context.items():
    #     # Check if the value is scalar (shape == ()) and handle accordingly
    #     if value.shape == ():
    #         print(f"  {key}: scalar - {value.numpy()}")
    #     else:
    #         print(f"  {key}: {value.shape} points ")# - {value.numpy()}")  # For non-scalar values

    # print("\nParsed Features:")
    # for key, value in parsed_features.items():
    #     # Check if the value is scalar (shape == ()) and handle accordingly
    #     if value.shape == ():
    #         print(f"  {key}: scalar - {value.numpy()}")
    #     else:
    #         print(f"  {key}: {value.shape} points ")#- {value.numpy() if isinstance(value, tf.Tensor) else value}")

    sim_array = []
    positions_all_t = parsed_features['position']

    positions_all_numpy = positions_all_t.numpy()
    positions_all_t_torch = torch.tensor(positions_all_numpy)

    total_array.append(positions_all_t_torch)
    # print(positions_all_t_torch.shape)
    # print(positions_all_t[0])
    # break

    # for t in range(len(positions_all_t)): # iterates over timesteps
    #   positions = positions_all_t[t]
    #   positions = tf.convert_to_tensor(positions, dtype=tf.float32) 

    #   # Step 1: Convert to NumPy array
    #   np_positions_array = positions.numpy()

    #   # Step 2: Convert to PyTorch tensor
    #   torch_positions_array = torch.from_numpy(np_positions_array)
    #   sim_array.append(torch_positions_array)
    
    #   # print(len(sim_array))
    #   # print(type(sim_array[0]))
    # torch_sim_array = torch.stack(sim_array)

    # total_array.append(torch_sim_array)

    # print(torch_sim_array)
    # break
  #print(total_array)


print(len(total_array))
print(total_array[0].shape)
torch.save(total_array, './tensorFiles/position_tensor_test.pt')