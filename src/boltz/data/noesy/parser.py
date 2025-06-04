import collections
import re
from typing import Dict, List, Any, Optional

class NOESYParser:
    """
    Parses NOESY data files.
    The expected format for each line is:
    "residueFrom residueTo peakID distance atomFrom atomTo"
    Example: "A101 B202 1 2.5 HA HN"
    Residue identifiers like "A101" will be parsed to extract the numeric part (101).
    """

    def __init__(self, max_residues: Optional[int] = None):
        """
        Initializes the NOESYParser.

        Args:
            max_residues (Optional[int]): Not currently used, but could be used for
                                          filtering or padding data based on sequence length.
        """
        self.max_residues = max_residues

    def _parse_residue_id(self, res_id_str: str) -> int:
        """
        Parses a residue identifier string (e.g., "A101", "101") to an integer.
        Converts 1-based residue number to 0-based index.
        """
        match = re.search(r'\d+', res_id_str)
        if match:
            return int(match.group(0)) - 1  # Convert to 0-based
        else:
            raise ValueError(f"Could not parse residue number from {res_id_str}")

    def parse(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parses a NOESY data file.

        Args:
            file_path (str): Path to the NOESY data file.
            sequence_length (int): The length of the protein sequence. This argument
                                   was in the prompt but seems more relevant for featurization.
                                   It's not strictly needed for parsing if we just store raw parsed data.
                                   I will remove it from here and ensure it's used in the featurizer.


        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing:
                - 'peaks': A list of all parsed peak assignments. Each assignment is a dictionary:
                           {'res_from': int, 'res_to': int, 'peak_id': int,
                            'distance': float, 'atom_from': str, 'atom_to': str,
                            'res_from_full': str, 'res_to_full': str}
                           'res_from' and 'res_to' are 0-based indices.
                - 'grouped_by_peak_id': A dictionary where keys are peak_ids (int) and
                                       values are lists of peak assignment dictionaries
                                       belonging to that peak_id. This structure helps
                                       identify ambiguous peaks.
        """
        parsed_peaks: List[Dict[str, Any]] = []
        grouped_by_peak_id: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)

        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) != 6:
                        print(f"Warning: Skipping malformed line {line_num} in {file_path}: {line}. Expected 6 parts, got {len(parts)}")
                        continue

                    res_from_str, res_to_str, peak_id_str, dist_str, atom_from, atom_to = parts

                    try:
                        res_from_0based = self._parse_residue_id(res_from_str)
                        res_to_0based = self._parse_residue_id(res_to_str)
                        peak_id = int(peak_id_str)
                        distance = float(dist_str)
                    except ValueError as e:
                        print(f"Warning: Skipping line {line_num} in {file_path} due to data conversion error: {e}. Line: {line}")
                        continue

                    peak_data = {
                        'res_from': res_from_0based,
                        'res_to': res_to_0based,
                        'peak_id': peak_id,
                        'distance': distance,
                        'atom_from': atom_from,
                        'atom_to': atom_to,
                        'res_from_full': res_from_str, # Keep original string for reference
                        'res_to_full': res_to_str,   # Keep original string for reference
                    }
                    parsed_peaks.append(peak_data)
                    grouped_by_peak_id[peak_id].append(peak_data)

        except FileNotFoundError:
            print(f"Error: File not found {file_path}")
            # Depending on desired behavior, could raise error or return empty
            return {'peaks': [], 'grouped_by_peak_id': {}}
        except Exception as e:
            print(f"An unexpected error occurred while parsing {file_path}: {e}")
            return {'peaks': [], 'grouped_by_peak_id': {}}

        return {'peaks': parsed_peaks, 'grouped_by_peak_id': dict(grouped_by_peak_id)}

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Create a dummy NOESY file
    dummy_file_content = """
    # This is a comment
    A1 B5 1 2.5 HA HN
    GLY20 LYS30 2 3.0 H H
    A1 C10 1 3.5 HB CA  # Ambiguous with peak 1
    ARG50 ALA60 3 4.0 N N
    VAL101 ILE150 4 1.8 HG1 HG2
    Xaa10 Y20 5 invalid_dist H H # Invalid distance
    A5 B6 6 2.2 H N N # Malformed line
    """
    dummy_file_path = "dummy_noesy.txt"
    with open(dummy_file_path, "w") as f:
        f.write(dummy_file_content)

    parser = NOESYParser()
    parsed_data = parser.parse(dummy_file_path)

    print("All Parsed Peaks:")
    for peak in parsed_data['peaks']:
        print(peak)

    print("\nGrouped by Peak ID (Ambiguous Peaks):")
    for peak_id, assignments in parsed_data['grouped_by_peak_id'].items():
        if len(assignments) > 1:
            print(f"Peak ID {peak_id}:")
            for assignment in assignments:
                print(f"  {assignment}")

    # Clean up dummy file
    import os
    os.remove(dummy_file_path)
