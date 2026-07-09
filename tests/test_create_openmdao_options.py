"""Unit tests for create_openmdao_options in om_converter.py.

Tests verify that list-valued options are converted to tuples,
non-list values pass through unchanged, and aviary_options
with __aviary_values__ are handled correctly.

Requirements: 2.1, 2.2, 2.3
"""

from unittest.mock import patch, MagicMock

from standard_evaluator.om_converter import create_openmdao_options


class TestCreateOpenmdaoOptionsSingleElementList:
    """Test list-to-tuple conversion for single-element lists (e.g., default_shape: [3])."""

    def test_single_element_list_converted_to_tuple(self):
        """A single-element list like [3] should become the tuple (3,).

        Validates: Requirements 2.1, 2.2
        """
        info_dict = {
            '_dict': {
                'default_shape': {'val': [3]},
            }
        }
        result = create_openmdao_options(info_dict)
        assert result['default_shape'] == (3,)
        assert isinstance(result['default_shape'], tuple)


class TestCreateOpenmdaoOptionsMultiElementList:
    """Test list-to-tuple conversion for multi-element lists (e.g., default_shape: [2, 4])."""

    def test_multi_element_list_converted_to_tuple(self):
        """A multi-element list like [2, 4] should become the tuple (2, 4).

        Validates: Requirements 2.1, 2.2
        """
        info_dict = {
            '_dict': {
                'default_shape': {'val': [2, 4]},
            }
        }
        result = create_openmdao_options(info_dict)
        assert result['default_shape'] == (2, 4)
        assert isinstance(result['default_shape'], tuple)


class TestCreateOpenmdaoOptionsNonListValues:
    """Test that non-list values (strings, ints) pass through unchanged."""

    def test_string_value_passes_through(self):
        """String option values should not be altered.

        Validates: Requirements 2.3
        """
        info_dict = {
            '_dict': {
                'units': {'val': 'kg'},
            }
        }
        result = create_openmdao_options(info_dict)
        assert result['units'] == 'kg'
        assert isinstance(result['units'], str)

    def test_int_value_passes_through(self):
        """Integer option values should not be altered.

        Validates: Requirements 2.3
        """
        info_dict = {
            '_dict': {
                'num_nodes': {'val': 10},
            }
        }
        result = create_openmdao_options(info_dict)
        assert result['num_nodes'] == 10
        assert isinstance(result['num_nodes'], int)

    def test_mixed_options(self):
        """A dict with both list and non-list values should convert only lists.

        Validates: Requirements 2.1, 2.2, 2.3
        """
        info_dict = {
            '_dict': {
                'default_shape': {'val': [3, 5]},
                'units': {'val': 'm/s'},
                'num_nodes': {'val': 7},
            }
        }
        result = create_openmdao_options(info_dict)
        assert result['default_shape'] == (3, 5)
        assert isinstance(result['default_shape'], tuple)
        assert result['units'] == 'm/s'
        assert result['num_nodes'] == 7


class TestCreateOpenmdaoOptionsAviaryValues:
    """Test handling of aviary_options with __aviary_values__ key."""

    @patch('standard_evaluator.om_converter.convert_aviary')
    def test_aviary_options_with_aviary_values_calls_convert(self, mock_convert):
        """When aviary_options contains __aviary_values__, convert_aviary is called.

        Validates: Requirements 2.3
        """
        mock_aviary_result = MagicMock(name='AviaryValues')
        mock_convert.return_value = mock_aviary_result

        aviary_data = {'some_key': 'some_value'}
        info_dict = {
            '_dict': {
                'aviary_options': {'val': {'__aviary_values__': aviary_data}},
                'default_shape': {'val': [2]},
            }
        }
        result = create_openmdao_options(info_dict)

        mock_convert.assert_called_once_with(aviary_data)
        assert result['aviary_options'] is mock_aviary_result
        assert result['default_shape'] == (2,)

    def test_aviary_options_without_aviary_values_is_deleted(self):
        """When aviary_options does NOT contain __aviary_values__, it is removed.

        Validates: Requirements 2.3
        """
        info_dict = {
            '_dict': {
                'aviary_options': {'val': {'other_key': 'data'}},
                'units': {'val': 'ft'},
            }
        }
        result = create_openmdao_options(info_dict)
        assert 'aviary_options' not in result
        assert result['units'] == 'ft'
