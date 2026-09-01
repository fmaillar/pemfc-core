import numpy as np
import pytest

from pemfc.src.channel import Channel, IncompressibleFluidChannel
from pemfc.src.fluid.species import PhaseChangeProperties
from pemfc.src.global_functions import add_source


@pytest.mark.parametrize(
    ('direction', 'expected'),
    [
        (1, [10.0, 11.0, 13.0, 16.0]),
        (-1, [16.0, 15.0, 13.0, 10.0]),
    ],
)
def test_discrete_sources_preserve_flow_balance(direction, expected):
    inlet_profile = np.full(4, 10.0)
    sources = np.array([1.0, 2.0, 3.0])

    result, applied_sources = add_source(
        inlet_profile, sources, direction=direction)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(applied_sources, sources)
    inlet = result[0] if direction == 1 else result[-1]
    outlet = result[-1] if direction == 1 else result[0]
    assert outlet == pytest.approx(inlet + applied_sources.sum())


def test_flow_limiter_preserves_balance_when_sources_would_go_negative():
    result, applied_sources = add_source(
        np.full(4, 5.0), np.array([-3.0, -4.0, 2.0]), direction=1)

    np.testing.assert_array_equal(result, [5.0, 2.0, 0.0, 2.0])
    np.testing.assert_array_equal(applied_sources, [-3.0, -2.0, 2.0])
    assert result[-1] == pytest.approx(result[0] + applied_sources.sum())
    assert np.all(result >= 0.0)


def test_reverse_flow_limiter_preserves_balance():
    result, applied_sources = add_source(
        np.full(4, 5.0), np.array([2.0, -4.0, -3.0]), direction=-1)

    np.testing.assert_array_equal(result, [2.0, 0.0, 2.0, 5.0])
    np.testing.assert_array_equal(applied_sources, [2.0, -2.0, -3.0])
    assert result[0] == pytest.approx(result[-1] + applied_sources.sum())
    assert np.all(result >= 0.0)


def test_reverse_mix_temperature_preserves_local_enthalpy_balance():
    channel = object.__new__(IncompressibleFluidChannel)
    channel.temp_ele = np.zeros(2)
    channel._flow_direction = -1
    channel.g_fluid = np.array([2.0, 4.0, 8.0])
    initial_temperature = np.array([0.0, 0.0, 300.0])
    channel.temperature = initial_temperature.copy()
    channel.heat = np.array([20.0, 40.0])
    enthalpy_source = np.array([20.0, 40.0])

    Channel.calc_mix_temperature(channel, enthalpy_source)

    expected_enthalpy = (
        channel.g_fluid[1:] * initial_temperature[1:]
        + channel.heat
        + enthalpy_source
    )
    np.testing.assert_allclose(
        channel.g_fluid[:-1] * channel.temperature[:-1],
        expected_enthalpy,
    )


def test_humidification_preserves_total_molar_fraction_and_dry_ratio():
    properties = PhaseChangeProperties({'H2O': object()})
    dry_composition = np.array([0.21, 0.79, 0.0])
    original_composition = dry_composition.copy()
    humidity = 0.5
    temperature = 333.15
    pressure = 101325.0

    humid_composition = properties.calc_humid_composition(
        humidity, temperature, pressure, dry_composition, id_pc=2)

    expected_water_fraction = float(
        humidity
        * properties.calc_saturation_pressure(temperature)[0]
        / pressure
    )
    assert humid_composition.sum() == pytest.approx(1.0)
    assert humid_composition[2] == pytest.approx(expected_water_fraction)
    assert humid_composition[0] / humid_composition[1] == pytest.approx(
        dry_composition[0] / dry_composition[1]
    )
    np.testing.assert_array_equal(dry_composition, original_composition)


def test_humidity_above_saturation_is_rejected():
    properties = PhaseChangeProperties({'H2O': object()})

    with pytest.raises(ValueError, match='relative humidity'):
        properties.calc_humid_composition(
            1.01, 333.15, 101325.0, np.array([0.21, 0.79, 0.0]), id_pc=2
        )
