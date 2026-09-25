"""Unit tests for workload_mix.py (the WORKLOAD_MIXES parser used by the MoST harness).

The module under test lives in the repository root (next to experiment_automation.py), so the
repository root is added to sys.path before importing it. Run with:
    pytest fmperf/tests/test_workload_mix.py
or
    python fmperf/tests/test_workload_mix.py
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workload_mix import mix_spec_payload, output_bounds_for_input, parse_workload_mixes


def _filename_factory(in_min, in_max):
    """Stand-in for the harness filename resolver used in the automation."""
    return f'sample_requests_{in_min}-{in_max}.json'


class TestParseWorkloadMixes(unittest.TestCase):
    def test_empty_value_yields_no_mixes(self):
        self.assertEqual(parse_workload_mixes(''), [])
        self.assertEqual(parse_workload_mixes(None), [])

    def test_single_mix_with_two_profiles(self):
        mixes = parse_workload_mixes('[(1-100:1-100,0.5), (300-600:100-300,0.5)]')
        self.assertEqual(len(mixes), 1)
        mix = mixes[0]
        self.assertEqual(len(mix['profiles']), 2)
        self.assertEqual(mix['canonical'], '[(1-100:1-100,0.5),(300-600:100-300,0.5)]')
        self.assertEqual(mix['parent_dir'], 'mix_1-100_1-100@0.5+300-600_100-300@0.5')
        self.assertEqual(
            mix['envelope'],
            {'in_min': 1, 'in_max': 600, 'out_min': 1, 'out_max': 300},
        )
        self.assertAlmostEqual(sum(p['alpha'] for p in mix['profiles']), 1.0)

    def test_multiple_mixes_are_independent(self):
        value = '[(1-100:1-100,0.5),(300-600:100-300,0.5)],[(1-100:1-100,0.25),(300-600:100-300,0.75)]'
        mixes = parse_workload_mixes(value)
        self.assertEqual(len(mixes), 2)
        self.assertEqual(mixes[0]['parent_dir'], 'mix_1-100_1-100@0.5+300-600_100-300@0.5')
        self.assertEqual(mixes[1]['parent_dir'], 'mix_1-100_1-100@0.25+300-600_100-300@0.75')

    def test_alphas_are_normalised(self):
        mixes = parse_workload_mixes('[(1-100:1-100,1),(300-600:100-300,3)]')
        alphas = [p['alpha'] for p in mixes[0]['profiles']]
        self.assertAlmostEqual(alphas[0], 0.25)
        self.assertAlmostEqual(alphas[1], 0.75)
        self.assertEqual(mixes[0]['canonical'], '[(1-100:1-100,0.25),(300-600:100-300,0.75)]')

    def test_duplicate_profiles_in_one_mix_are_summed(self):
        mixes = parse_workload_mixes('[(1-100:1-100,0.25),(1-100:1-100,0.25),(300-600:100-300,0.5)]')
        mix = mixes[0]
        self.assertEqual(len(mix['profiles']), 2)
        self.assertAlmostEqual(mix['profiles'][0]['alpha'], 0.5)
        self.assertEqual(mix['parent_dir'], 'mix_1-100_1-100@0.5+300-600_100-300@0.5')

    def test_single_token_values_are_normalised_to_ranges(self):
        mixes = parse_workload_mixes('[(32:64,1)]')
        profile = mixes[0]['profiles'][0]
        self.assertEqual(profile['label'], '32-32:64-64')
        self.assertEqual((profile['in_min'], profile['in_max']), (32, 32))
        self.assertEqual((profile['out_min'], profile['out_max']), (64, 64))

    def test_reversed_bounds_are_swapped(self):
        profile = parse_workload_mixes('[(600-300:200-100,1)]')[0]['profiles'][0]
        self.assertEqual((profile['in_min'], profile['in_max']), (300, 600))
        self.assertEqual((profile['out_min'], profile['out_max']), (100, 200))

    def test_semicolon_separator_is_accepted(self):
        mixes = parse_workload_mixes('[(1-100:1-100,0.5);(300-600:100-300,0.5)]')
        self.assertEqual(len(mixes[0]['profiles']), 2)

    def test_malformed_pairs_are_skipped_without_losing_the_mix(self):
        mixes = parse_workload_mixes(
            '[(1-100:1-100,0.5),(no-colon,0.5),(300-600:100-300,not-a-number),(1-100:1-100,0)]'
        )
        self.assertEqual(len(mixes), 1)
        self.assertEqual(len(mixes[0]['profiles']), 1)
        self.assertEqual(mixes[0]['profiles'][0]['label'], '1-100:1-100')

    def test_mix_without_valid_pairs_is_skipped(self):
        self.assertEqual(parse_workload_mixes('[(garbage)]'), [])
        self.assertEqual(parse_workload_mixes('[(1-100,0.5)]'), [])

    def test_duplicate_mixes_are_ignored(self):
        value = '[(1-100:1-100,0.5),(300-600:100-300,0.5)],[(1-100:1-100,0.5),(300-600:100-300,0.5)]'
        self.assertEqual(len(parse_workload_mixes(value)), 1)

    def test_unbracketed_value_is_ignored(self):
        self.assertEqual(parse_workload_mixes('1-100:1-100,0.5'), [])

    def test_zero_boundaries_are_rejected(self):
        # The loadgen requires positive token bounds, so a 0 boundary is skipped with a warning.
        self.assertEqual(parse_workload_mixes('[(1-100:0-100,1)]'), [])
        self.assertEqual(parse_workload_mixes('[(0-100:1-100,1)]'), [])
        mixes = parse_workload_mixes('[(1-100:0-100,0.5),(300-600:100-300,0.5)]')
        self.assertEqual(len(mixes[0]['profiles']), 1)
        self.assertEqual(mixes[0]['profiles'][0]['label'], '300-600:100-300')



class TestMixSpecPayload(unittest.TestCase):
    def test_payload_carries_profile_routing_information(self):
        mixes = parse_workload_mixes('[(1-100:1-100,0.5),(300-600:100-300,0.5)]')
        payload = json.loads(mix_spec_payload(mixes[0], _filename_factory))
        self.assertEqual(payload['mix'], '[(1-100:1-100,0.5),(300-600:100-300,0.5)]')
        self.assertEqual(len(payload['profiles']), 2)
        first = payload['profiles'][0]
        self.assertEqual(first['label'], '1-100:1-100')
        self.assertEqual(first['filename'], 'sample_requests_1-100.json')
        self.assertEqual((first['out_min'], first['out_max']), (1, 100))
        self.assertAlmostEqual(first['alpha'], 0.5)
        second = payload['profiles'][1]
        self.assertEqual(second['filename'], 'sample_requests_300-600.json')

    def test_profiles_sharing_an_input_interval_share_the_file(self):
        mixes = parse_workload_mixes('[(1-100:1-100,0.5),(1-100:100-300,0.5)]')
        payload = json.loads(mix_spec_payload(mixes[0], _filename_factory))
        filenames = {p['filename'] for p in payload['profiles']}
        self.assertEqual(filenames, {'sample_requests_1-100.json'})


class TestOutputBoundsForInput(unittest.TestCase):
    def test_union_across_mixes_for_the_same_input_interval(self):
        mixes = parse_workload_mixes(
            '[(1-100:1-100,0.5),(300-600:100-300,0.5)],[(1-100:600-1000,1)]'
        )
        self.assertEqual(output_bounds_for_input(mixes, 1, 100), (1, 1000))
        self.assertEqual(output_bounds_for_input(mixes, 300, 600), (100, 300))

    def test_fallback_when_no_profile_matches(self):
        mixes = parse_workload_mixes('[(1-100:1-100,1)]')
        self.assertEqual(output_bounds_for_input(mixes, 200, 300), (200, 300))
        self.assertEqual(
            output_bounds_for_input(mixes, 200, 300, fallback=(1, 5)), (1, 5)
        )

    def test_lower_bound_is_the_minimum_of_matching_profiles(self):
        mixes = parse_workload_mixes('[(1-100:600-1000,0.5),(1-100:100-300,0.5)]')
        self.assertEqual(output_bounds_for_input(mixes, 1, 100), (100, 1000))


if __name__ == '__main__':
    unittest.main()
