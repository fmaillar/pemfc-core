import numpy as np
import pytest

from pemfc.src.channel import Channel, IncompressibleFluidChannel
from pemfc.src.output_object import OutputObject1D
from pemfc.src.transport_layer import TransportLayer, TransportLayer3D


class OutputFixture(OutputObject1D):
    pass


def test_serial_conductance_uses_both_inputs():
    result = TransportLayer.connect(
        np.array([2.0, 1.0]), np.array([4.0, 3.0]), mode='serial')

    np.testing.assert_allclose(result, np.array([4.0 / 3.0, 0.75]))


def test_serial_conductance_is_zero_if_either_input_is_zero():
    result = TransportLayer.connect(
        np.array([2.0, 0.0]), np.array([0.0, 3.0]), mode='serial')

    np.testing.assert_array_equal(result, np.zeros(2))


def test_reduce_conductance_updates_each_property():
    layer = object.__new__(TransportLayer3D)
    layer.conductance = {
        'thermal': np.ones((2, 2, 2)),
        'electrical': np.full((2, 2, 2), 2.0),
    }

    layer.reduce_conductance(0.5, indices=0, axis=0)

    np.testing.assert_array_equal(layer.conductance['thermal'][0], 0.5)
    np.testing.assert_array_equal(layer.conductance['electrical'][0], 1.0)
    np.testing.assert_array_equal(layer.conductance['thermal'][1], 1.0)


def test_relative_plot_axis_does_not_accumulate_between_variables():
    output = OutputFixture('fixture')
    output.add_print_data(np.array([1.0]), 'first', '-')
    output.add_print_data(np.array([2.0]), 'second', '-')

    output.set_plot_axis(2, relative=True)

    assert output.single_print_data['first']['plot_axis'] == 1
    assert output.single_print_data['second']['plot_axis'] == 1


def test_print_variables_resolve_public_nested_attributes():
    output = OutputFixture('fixture')
    output.values = np.array([1.0, 2.0])
    output.labels = ['first', 'second']

    output.add_print_variables({
        'names': ['values'],
        'units': ['-'],
        'sub_names': ['self.labels'],
    })

    assert list(output.multi_print_data['Values']) == output.labels


@pytest.mark.parametrize(
    'path',
    ["__class__", "values[0]", "values.__class__", "open('/tmp/file')"],
)
def test_print_variable_paths_reject_expressions_and_private_attributes(path):
    output = OutputFixture('fixture')
    output.values = np.array([1.0])

    with pytest.raises(ValueError, match='attribute path'):
        output.add_print_variables({
            'names': [path],
            'units': ['-'],
            'sub_names': ['None'],
        })


def test_forward_mix_temperature_uses_local_fluid_capacity_rate():
    channel = object.__new__(IncompressibleFluidChannel)
    channel.temp_ele = np.zeros(2)
    channel._flow_direction = 1
    channel.g_fluid = np.array([2.0, 4.0, 8.0])
    channel.temperature = np.array([300.0, 0.0, 0.0])
    channel.heat = np.zeros(2)

    Channel.calc_mix_temperature(channel, np.array([40.0, 80.0]))

    np.testing.assert_allclose(channel.temperature, [300.0, 160.0, 10.0])
