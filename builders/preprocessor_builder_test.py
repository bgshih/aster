import tensorflow as tf

from google.protobuf import text_format

from rare.core import preprocessor
from rare.core import standard_fields as fields
from rare.protos import preprocessor_pb2
from rare.builders import preprocessor_builder

class PreprocessorBuilderTest(tf.test.TestCase):

  def assert_dictionary_close(self, dict1, dict2):
    """Helper to check if two dicts with floatst or integers are close."""
    self.assertEqual(sorted(dict1.keys()), sorted(dict2.keys()))
    for key in dict1:
      value = dict1[key]
      if isinstance(value, float):
        self.assertAlmostEqual(value, dict2[key])
      elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], float):
        self.assertAllClose(value, dict2[key])
      elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], int):
        self.assertEqual(value, dict2[key])
      else:
        self.assertEqual(value, dict2[key])

  def test_build_resize_image_random_method(self):
    preprocessor_text_proto = """
    resize_image_random_method {
      target_height: 384
      target_width: 384
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.resize_image_random_method)
    self.assert_dictionary_close(args, {
        'target_size': [384, 384]
    })

  def test_build_resize_image(self):
    preprocessor_text_proto = """
    resize_image {
      target_height: 384
      target_width: 384
      method: BICUBIC
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.resize_image)
    self.assertEqual(args, {
        'target_size': [384, 384],
        'method': tf.image.ResizeMethod.BICUBIC
    })

  def test_normalize_image(self):
    preprocessor_text_proto = """
    normalize_image {
      original_minval: 0.0
      original_maxval: 255.0
      target_minval: 0.0
      target_maxval: 1.0
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.normalize_image)
    self.assert_dictionary_close(args, {
        'original_minval': 0.0,
        'original_maxval': 255.0,
        'target_minval': 0.0,
        'target_maxval': 1.0
    })

  def test_random_pixel_value_scale(self):
    preprocessor_text_proto = """
    random_pixel_value_scale {
      minval: 0.85
      maxval: 1.25
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_pixel_value_scale)
    self.assert_dictionary_close(args, {
        'minval': 0.85,
        'maxval': 1.25
    })

  def test_random_rgb_to_gray(self):
    preprocessor_text_proto = """
    random_rgb_to_gray {
      probability: 0.15
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_rgb_to_gray)
    self.assert_dictionary_close(args, {
        'probability': 0.15
    })

  def test_random_adjust_brightness(self):
    preprocessor_text_proto = """
    random_adjust_brightness {
      max_delta: 0.15
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_adjust_brightness)
    self.assert_dictionary_close(args, {
        'max_delta': 0.15
    })
  
  def test_random_adjust_contrast(self):
    preprocessor_text_proto = """
    random_adjust_contrast {
      min_delta: 0.7
      max_delta: 1.3
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_adjust_contrast)
    self.assert_dictionary_close(args, {
        'min_delta': 0.7,
        'max_delta': 1.3
    })

  def test_string_filtering(self):
    preprocessor_text_proto = """
    string_filtering {
      lower_case: true
      include_charset: "abc"
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.string_filtering)
    self.assert_dictionary_close(args, {
        'lower_case': True,
        'include_charset': "abc"
    })

    test_input_strings = [t.encode('utf-8') for t in ['abc', 'abcde', 'ABCDE']]
    expected_output_string = [t.encode('utf-8') for t in ['abc', 'abc', 'abc']]
    test_processed_strings = [function(t, **args) for t in test_input_strings]
    with self.test_session() as sess:
      outputs = sess.run(test_processed_strings)
      self.assertAllEqual(outputs, expected_output_string)

  def test_string_filtering_2(self):
    preprocessor_text_proto = """
    string_filtering {
      lower_case: false
      include_charset: "abcdABCD"
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.string_filtering)
    self.assert_dictionary_close(args, {
        'lower_case': False,
        'include_charset': "abcdABCD"
    })

    test_input_strings = [t.encode('utf-8') for t in ['abc', 'abcde', '!=ABC DE~']]
    expected_output_string = [t.encode('utf-8') for t in ['abc', 'abcd', 'ABCD']]
    test_processed_strings = [function(t, **args) for t in test_input_strings]
    with self.test_session() as sess:
      outputs = sess.run(test_processed_strings)
      self.assertAllEqual(outputs, expected_output_string)


if __name__ == '__main__':
  tf.test.main()
