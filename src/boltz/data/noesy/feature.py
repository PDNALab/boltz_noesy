import torch
import numpy as np
from typing import Dict, List, Any
from .parser import NOESYParser # Assuming parser.py is in the same directory

class NOESYFeature:
    """
    Generates 2D feature tensors from parsed NOESY data.
    The primary feature is a one-hot encoded discretized distance.
    """

    def __init__(self, num_bins: int = 64, min_dist: float = 0.0, max_dist: float = 5.0,
                 noise_threshold: Optional[float] = None):
        """
        Initializes the NOESYFeature.

        Args:
            num_bins (int): Number of bins for discretizing distances.
            min_dist (float): Minimum distance for the first bin.
            max_dist (float): Maximum distance for the last bin (upper edge of the last bin).
            noise_threshold (Optional[float]): A parameter to potentially filter out noisy peaks
                                     (e.g., based on distance). Not actively used in current
                                     featurization if only shortest distance is chosen.
        """
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.noise_threshold = noise_threshold # Not used in this version

        # Define bin edges: num_bins + 1 edges for num_bins
        # Ensure max_dist is included in the last bin properly.
        # Linspace creates num_bins points, we need num_bins+1 edges for np.digitize/torch.bucketize
        self.bin_edges = torch.linspace(min_dist, max_dist, num_bins + 1)


    def featurize(self, parsed_noesy_data: Dict[str, List[Dict[str, Any]]], sequence_length: int) -> torch.Tensor:
        """
        Creates a 2D feature tensor from parsed NOESY data.
        The feature tensor shape is (sequence_length, sequence_length, num_bins).
        It represents one-hot encoded binned distances. If multiple peaks connect
        the same (i,j) residue pair, the one with the shortest distance is used.

        Args:
            parsed_noesy_data (Dict[str, List[Dict[str, Any]]]):
                The output from NOESYParser.parse(). Expects a dictionary with a 'peaks' key,
                which is a list of peak assignment dictionaries.
            sequence_length (int): The length of the protein sequence.

        Returns:
            torch.Tensor: A tensor of shape (sequence_length, sequence_length, num_bins)
                          with dtype torch.float32. feature_tensor[i, j, k] = 1 if the
                          shortest distance between residue i and j falls into bin k.
        """
        feature_tensor = torch.zeros((sequence_length, sequence_length, self.num_bins), dtype=torch.float32)

        if not parsed_noesy_data or 'peaks' not in parsed_noesy_data or not parsed_noesy_data['peaks']:
            return feature_tensor

        # Store shortest distances found for each pair (i,j)
        # Key: (min_res_idx, max_res_idx), Value: shortest_distance
        min_distances_map: Dict[tuple[int, int], float] = {}

        for peak in parsed_noesy_data['peaks']:
            res_from = peak['res_from']
            res_to = peak['res_to']
            distance = peak['distance']

            # Validate residue indices and ensure they are different
            if res_from == res_to:
                continue
            if not (0 <= res_from < sequence_length and 0 <= res_to < sequence_length):
                print(f"Warning: Residue index out of bounds (0-{sequence_length-1}). "
                      f"res_from: {res_from}, res_to: {res_to}. Skipping peak.")
                continue

            # Filter by distance if noise_threshold is set (e.g. very long distances)
            if self.noise_threshold is not None and distance > self.noise_threshold:
                continue

            # Ensure consistent key for pairs (i,j) and (j,i)
            pair_key = tuple(sorted((res_from, res_to)))

            if pair_key not in min_distances_map or distance < min_distances_map[pair_key]:
                min_distances_map[pair_key] = distance

        # Populate the feature tensor with binned shortest distances
        for (r1, r2), dist_val in min_distances_map.items():
            # Ensure distance is within the chosen range [min_dist, max_dist]
            # torch.bucketize will place values outside range into 0 or len(self.bin_edges)-1
            # We want to assign to a bin k, where self.bin_edges[k] <= value < self.bin_edges[k+1]
            # So, values equal to max_dist should go into the last bin (num_bins-1)
            # torch.bucketize(..., right=True) helps: bin_edges[i-1] < x <= bin_edges[i]
            # If we use default (right=False): bin_edges[i] <= x < bin_edges[i+1]

            # Clamp distance to be within [min_dist, max_dist] before binning
            # to avoid issues with values outside the defined bin_edges.
            clamped_dist = np.clip(dist_val, self.min_dist, self.max_dist)

            # `torch.bucketize` returns 1-based indexing if `self.bin_edges` are used directly
            # and value is equal to the first edge. Or can be 0 if less than first edge.
            # It's easier to think of bins as 0 to num_bins-1.
            # If clamped_dist == self.min_dist, it should be bin 0.
            # If clamped_dist == self.max_dist, it should be bin num_bins-1.

            # Subtracting self.min_dist and dividing by bin_width is more robust for direct bin index calc
            # bin_width = (self.max_dist - self.min_dist) / self.num_bins
            # bin_index = int((clamped_dist - self.min_dist) / bin_width)
            # this can cause issues if clamped_dist == self.max_dist, leading to bin_index == self.num_bins

            # Using torch.bucketize:
            # self.bin_edges has N+1 elements.
            # if clamped_dist is self.max_dist, bucketize might put it into bin N if not careful.
            # We need bin indices from 0 to N-1.
            bin_index = torch.bucketize(torch.tensor(clamped_dist), self.bin_edges, right=False)

            # Adjust bin_index:
            # If value == min_dist, bucketize gives 0 if bin_edges[0] == min_dist
            # If value == max_dist, bucketize(max_dist, [e0,e1,...,eN], right=False) can give N.
            # We want it in bin N-1.
            if bin_index >= self.num_bins: # If it fell into the "overflow" bin (i.e. exactly max_dist or slightly over due to float)
                 bin_index = self.num_bins - 1
            # If it's less than min_dist (should not happen due to clamping, but for safety)
            if bin_index < 0: # Should not happen with clamping and default bucketize
                bin_index = 0

            # Make sure the calculated bin_index is valid for one-hot encoding
            if 0 <= bin_index < self.num_bins:
                feature_tensor[r1, r2, bin_index] = 1.0
                feature_tensor[r2, r1, bin_index] = 1.0 # Symmetric

        return feature_tensor

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # 1. Setup a dummy parser and parse dummy data
    parser = NOESYParser()
    dummy_file_content = """
    A1 B5 1 2.5 HA HN
    A1 B5 2 2.3 HA HN  # Shorter distance for A1-B5
    C10 D12 3 0.5 H H
    E20 F22 4 4.8 N N
    G30 H32 5 5.5 N N  # Outside max_dist if max_dist is 5.0
    A1 Z500 6 2.0 H H # Z500 out of sequence_length bounds
    B5 A1 7 2.4 HE HE # Duplicate of A1 B5 essentially
    """
    dummy_file_path = "dummy_feature_noesy.txt"
    with open(dummy_file_path, "w") as f:
        f.write(dummy_file_content)

    parsed_data = parser.parse(dummy_file_path)

    # 2. Initialize Featurizer
    # sequence_length should be appropriate for the dummy data (e.g., up to H32)
    # Residues: A1 (0), B5 (4), C10 (9), D12 (11), E20 (19), F22 (21), G30 (29), H32 (31)
    # Max residue index is 31, so sequence_length should be at least 32.
    sequence_length = 40
    num_bins = 5 # For simpler output checking: (0-1), (1-2), (2-3), (3-4), (4-5)
    min_dist = 0.0
    max_dist = 5.0

    featurizer = NOESYFeature(num_bins=num_bins, min_dist=min_dist, max_dist=max_dist)

    # 3. Featurize
    feature_matrix = featurizer.featurize(parsed_data, sequence_length)

    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Check some expected features
    # A1 (0) and B5 (4) should have shortest distance 2.3. Bin for 2.3?
    # Bin edges: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # 2.3 is in bin 2 (0-indexed: edges[2] <= 2.3 < edges[3] => 2.0 <= 2.3 < 3.0)
    idx_A1 = parser._parse_residue_id("A1") # 0
    idx_B5 = parser._parse_residue_id("B5") # 4

    print(f"A1 index: {idx_A1}, B5 index: {idx_B5}")
    if idx_A1 < sequence_length and idx_B5 < sequence_length:
        print(f"Features for A1-B5 (0,4): {feature_matrix[idx_A1, idx_B5, :]}")
        expected_bin_A1_B5 = torch.bucketize(torch.tensor(2.3), featurizer.bin_edges, right=False)
        if expected_bin_A1_B5 >= num_bins: expected_bin_A1_B5 = num_bins -1
        print(f"Expected bin for 2.3A: {expected_bin_A1_B5}")
        if feature_matrix[idx_A1, idx_B5, expected_bin_A1_B5] == 1.0:
            print("A1-B5 shortest distance feature correctly placed.")
        else:
            print("Error: A1-B5 feature incorrect.")

    # C10 (9) and D12 (11) distance 0.5. Bin 0.
    idx_C10 = parser._parse_residue_id("C10") # 9
    idx_D12 = parser._parse_residue_id("D12") # 11
    print(f"C10 index: {idx_C10}, D12 index: {idx_D12}")
    if idx_C10 < sequence_length and idx_D12 < sequence_length:
        print(f"Features for C10-D12 (9,11): {feature_matrix[idx_C10, idx_D12, :]}")
        expected_bin_C10_D12 = torch.bucketize(torch.tensor(0.5), featurizer.bin_edges, right=False)
        if expected_bin_C10_D12 >= num_bins: expected_bin_C10_D12 = num_bins-1
        print(f"Expected bin for 0.5A: {expected_bin_C10_D12}")
        if feature_matrix[idx_C10, idx_D12, expected_bin_C10_D12] == 1.0:
            print("C10-D12 feature correctly placed.")
        else:
            print("Error: C10-D12 feature incorrect.")

    # G30 (29) and H32 (31) distance 5.5. Should be clamped to max_dist (5.0) and put in last bin (bin 4)
    idx_G30 = parser._parse_residue_id("G30") #29
    idx_H32 = parser._parse_residue_id("H32") #31
    print(f"G30 index: {idx_G30}, H32 index: {idx_H32}")
    if idx_G30 < sequence_length and idx_H32 < sequence_length:
        print(f"Features for G30-H32 (29,31): {feature_matrix[idx_G30, idx_H32, :]}")
        # Distance 5.5 is clamped to 5.0. Bin for 5.0 is num_bins-1
        expected_bin_G30_H32 = num_bins - 1
        print(f"Expected bin for 5.5A (clamped to 5.0A): {expected_bin_G30_H32}")
        if feature_matrix[idx_G30, idx_H32, expected_bin_G30_H32] == 1.0:
            print("G30-H32 feature correctly placed in last bin due to clamping.")
        else:
            print("Error: G30-H32 feature incorrect.")
            print(f"Bin edges: {featurizer.bin_edges}")
            clamped_val = np.clip(5.5, featurizer.min_dist, featurizer.max_dist)
            actual_b_idx = torch.bucketize(torch.tensor(clamped_val), featurizer.bin_edges, right=False)
            if actual_b_idx >= num_bins: actual_b_idx = num_bins -1
            print(f"Clamped val: {clamped_val}, actual bucketize index: {actual_b_idx}")


    # Check symmetry for A1-B5
    if torch.equal(feature_matrix[idx_A1, idx_B5, :], feature_matrix[idx_B5, idx_A1, :]):
        print("Symmetry confirmed for A1-B5.")
    else:
        print("Error: Symmetry failed for A1-B5.")
        print(f"{feature_matrix[idx_B5, idx_A1, :]}")

    # Clean up dummy file
    import os
    os.remove(dummy_file_path)
