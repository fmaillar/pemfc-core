import os
import json
import numpy as np
from main_app import main

# load settings
with open(os.path.join('tests', 'settings.json')) as file:
    ref_settings = json.load(file)

# load settings
with open(os.path.join('pemfc', 'settings', 'settings.json')) as file:
    main_settings = json.load(file)

# load reference results
with open(os.path.join('tests', 'summary.json')) as file:
    ref_results = json.load(file)


# def test_inputs():
#     assert main_settings == ref_settings


def test_global_results():
    g_data, l_data, sim = main(ref_settings)
    results = g_data[0]
    assert results['Convergence']['value']
    for name, reference in ref_results.items():
        assert results[name]['units'] == reference['units']
        np.testing.assert_allclose(
            results[name]['value'], reference['value'], rtol=1e-6)
