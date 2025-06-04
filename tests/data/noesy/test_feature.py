import unittest
import torch
import numpy as np

from boltz.data.noesy.feature import NOESYFeature

class TestNOESYFeature(unittest.TestCase):

    def test_featurize_basic(self):
        featurizer = NOESYFeature(num_bins=5, min_dist=0.0, max_dist=5.0)
        # Bin edges: [0, 1, 2, 3, 4, 5]
        # Bins: [0,1), [1,2), [2,3), [3,4), [4,5)]

        parsed_data = {
            'peaks': [
                {'res_from': 0, 'res_to': 1, 'distance': 2.5, 'peak_id': 1, 'atom_from': 'H1', 'atom_to': 'H2'}, # Bin 2
                {'res_from': 0, 'res_to': 2, 'distance': 0.5, 'peak_id': 2, 'atom_from': 'H1', 'atom_to': 'H3'}  # Bin 0
            ],
            'grouped_by_peak_id': {} # Not directly used by current featurizer logic
        }
        sequence_length = 3
        feature_tensor = featurizer.featurize(parsed_data, sequence_length)

        self.assertEqual(feature_tensor.shape, (sequence_length, sequence_length, 5))
        self.assertEqual(feature_tensor.dtype, torch.float32)

        # Check (0,1) contact -> distance 2.5 -> bin 2
        self.assertEqual(feature_tensor[0, 1, 2].item(), 1.0)
        self.assertEqual(feature_tensor[0, 1, :].sum().item(), 1.0) # One-hot
        self.assertEqual(feature_tensor[1, 0, 2].item(), 1.0) # Symmetric
        self.assertEqual(feature_tensor[1, 0, :].sum().item(), 1.0)

        # Check (0,2) contact -> distance 0.5 -> bin 0
        self.assertEqual(feature_tensor[0, 2, 0].item(), 1.0)
        self.assertEqual(feature_tensor[0, 2, :].sum().item(), 1.0)
        self.assertEqual(feature_tensor[2, 0, 0].item(), 1.0)
        self.assertEqual(feature_tensor[2, 0, :].sum().item(), 1.0)

        # Check other pairs are zero
        self.assertEqual(feature_tensor[1, 2, :].sum().item(), 0.0)


    def test_featurize_shortest_distance(self):
        featurizer = NOESYFeature(num_bins=5, min_dist=0.0, max_dist=5.0)
        parsed_data = {
            'peaks': [
                {'res_from': 0, 'res_to': 1, 'distance': 3.5, 'peak_id': 1, 'atom_from': 'H1', 'atom_to': 'H2'}, # Bin 3
                {'res_from': 0, 'res_to': 1, 'distance': 1.5, 'peak_id': 2, 'atom_from': 'H1', 'atom_to': 'H2'}, # Bin 1 (shorter)
                {'res_from': 0, 'res_to': 1, 'distance': 4.5, 'peak_id': 3, 'atom_from': 'H1', 'atom_to': 'H2'}  # Bin 4
            ],
            'grouped_by_peak_id': {}
        }
        sequence_length = 2
        feature_tensor = featurizer.featurize(parsed_data, sequence_length)

        # Shortest distance is 1.5, which is bin 1
        self.assertEqual(feature_tensor[0, 1, 1].item(), 1.0)
        self.assertEqual(feature_tensor[0, 1, :].sum().item(), 1.0)
        self.assertEqual(feature_tensor[1, 0, 1].item(), 1.0)

    def test_featurize_out_of_bounds_indices(self):
        featurizer = NOESYFeature(num_bins=5, min_dist=0.0, max_dist=5.0)
        parsed_data = {
            'peaks': [
                {'res_from': 0, 'res_to': 1, 'distance': 2.5, 'peak_id': 1, 'atom_from': 'H1', 'atom_to': 'H2'},
                {'res_from': 0, 'res_to': 3, 'distance': 1.5, 'peak_id': 2, 'atom_from': 'H1', 'atom_to': 'H2'} # res_to=3 is out of bounds
            ],
            'grouped_by_peak_id': {}
        }
        sequence_length = 2 # Max index is 1
        feature_tensor = featurizer.featurize(parsed_data, sequence_length)

        # Only (0,1) contact should be featurized
        self.assertEqual(feature_tensor[0, 1, 2].item(), 1.0)
        self.assertEqual(feature_tensor.sum().item(), 2.0) # 1.0 for (0,1) and 1.0 for (1,0)

    def test_featurize_no_contacts(self):
        featurizer = NOESYFeature(num_bins=5, min_dist=0.0, max_dist=5.0)
        parsed_data = {'peaks': [], 'grouped_by_peak_id': {}}
        sequence_length = 3
        feature_tensor = featurizer.featurize(parsed_data, sequence_length)

        self.assertEqual(feature_tensor.shape, (sequence_length, sequence_length, 5))
        self.assertTrue(torch.all(feature_tensor == 0.0))

    def test_featurize_distance_clamping(self):
        featurizer = NOESYFeature(num_bins=5, min_dist=1.0, max_dist=4.0)
        # Bin edges: [1, 1.6, 2.2, 2.8, 3.4, 4.0]
        # Bins: [1,1.6), [1.6,2.2), [2.2,2.8), [2.8,3.4), [3.4,4.0)]

        parsed_data = {
            'peaks': [
                {'res_from': 0, 'res_to': 1, 'distance': 0.5, 'peak_id': 1, 'atom_from': 'H1', 'atom_to': 'H2'},  # Below min_dist, clamped to 1.0 -> bin 0
                {'res_from': 0, 'res_to': 2, 'distance': 4.5, 'peak_id': 2, 'atom_from': 'H1', 'atom_to': 'H3'},  # Above max_dist, clamped to 4.0 -> bin 4
                {'res_from': 1, 'res_to': 2, 'distance': 4.0, 'peak_id': 3, 'atom_from': 'H1', 'atom_to': 'H3'}   # Equal to max_dist -> bin 4
            ],
            'grouped_by_peak_id': {}
        }
        sequence_length = 3
        feature_tensor = featurizer.featurize(parsed_data, sequence_length)

        # (0,1) distance 0.5 -> clamped to 1.0 -> bin 0
        self.assertEqual(feature_tensor[0, 1, 0].item(), 1.0)

        # (0,2) distance 4.5 -> clamped to 4.0 -> bin 4 (last bin)
        self.assertEqual(feature_tensor[0, 2, 4].item(), 1.0)

        # (1,2) distance 4.0 -> bin 4 (last bin)
        self.assertEqual(feature_tensor[1, 2, 4].item(), 1.0)

    def test_featurize_self_contact(self):
        featurizer = NOESYFeature(num_bins=5, min_dist=0.0, max_dist=5.0)
        parsed_data = {
            'peaks': [
                {'res_from': 0, 'res_to': 0, 'distance': 2.5, 'peak_id': 1, 'atom_from': 'H1', 'atom_to': 'H2'} # Self-contact
            ],
            'grouped_by_peak_id': {}
        }
        sequence_length = 2
        feature_tensor = featurizer.featurize(parsed_data, sequence_length)
        # Self-contacts should be ignored
        self.assertTrue(torch.all(feature_tensor == 0.0))

    def test_bin_edges_and_bucketize(self):
        num_bins = 64
        min_dist = 0.0
        max_dist = 5.0
        featurizer = NOESYFeature(num_bins=num_bins, min_dist=min_dist, max_dist=max_dist)

        # Check bin_edges
        self.assertEqual(len(featurizer.bin_edges), num_bins + 1)
        self.assertEqual(featurizer.bin_edges[0].item(), min_dist)
        self.assertTrue(np.isclose(featurizer.bin_edges[-1].item(), max_dist))

        # Test torch.bucketize behavior (as used in featurizer)
        # Values equal to min_dist should go to bin 0
        dist_min = torch.tensor(min_dist)
        bin_idx_min = torch.bucketize(dist_min, featurizer.bin_edges, right=False)
        if bin_idx_min >= num_bins: bin_idx_min = num_bins -1
        self.assertEqual(bin_idx_min.item(), 0)

        # Values equal to max_dist should go to bin num_bins-1
        dist_max = torch.tensor(max_dist)
        bin_idx_max = torch.bucketize(dist_max, featurizer.bin_edges, right=False)
        # if value == edge, it's placed in bin starting with that edge.
        # if max_dist is an edge, bucketize(max_dist, edges) can give len(edges)-1 which is num_bins.
        if bin_idx_max >= num_bins:
            bin_idx_max = num_bins - 1
        self.assertEqual(bin_idx_max.item(), num_bins - 1)

        # Value just under max_dist
        dist_slightly_less_max = torch.tensor(max_dist - 0.001)
        bin_idx_sl_max = torch.bucketize(dist_slightly_less_max, featurizer.bin_edges, right=False)
        if bin_idx_sl_max >= num_bins: bin_idx_sl_max = num_bins -1
        self.assertEqual(bin_idx_sl_max.item(), num_bins - 1)


if __name__ == '__main__':
    unittest.main()
