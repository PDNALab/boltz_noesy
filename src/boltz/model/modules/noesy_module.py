import torch
import torch.nn as nn
from typing import Optional

class NOESYModule(nn.Module):
    """
    A module to process NOESY features and project them to the pair representation dimension.
    """
    def __init__(
        self,
        noesy_feature_dim: int,
        token_z_dim: int, # Changed name to avoid clash with variable 'token_z'
        noesy_hidden_dim: Optional[int] = None
    ):
        """
        Args:
            noesy_feature_dim (int): The input dimension of NOESY features (e.g., num_bins).
            token_z_dim (int): The target dimension for pair representations.
            noesy_hidden_dim (Optional[int]): Optional intermediate dimension for an MLP.
                                              If None, a single Linear layer is used.
        """
        super().__init__()
        self.noesy_feature_dim = noesy_feature_dim
        self.token_z_dim = token_z_dim
        self.noesy_hidden_dim = noesy_hidden_dim

        if noesy_hidden_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(noesy_feature_dim, noesy_hidden_dim),
                nn.ReLU(),
                nn.Linear(noesy_hidden_dim, token_z_dim)
            )
        else:
            self.projection = nn.Linear(noesy_feature_dim, token_z_dim)

    def forward(self, noesy_feat: torch.Tensor) -> torch.Tensor:
        """
        Process NOESY features.

        Args:
            noesy_feat (torch.Tensor): NOESY features of shape
                                       (batch, seq_len, seq_len, noesy_feature_dim).

        Returns:
            torch.Tensor: Projected NOESY features of shape
                          (batch, seq_len, seq_len, token_z_dim).
        """
        if noesy_feat.shape[-1] != self.noesy_feature_dim:
            raise ValueError(
                f"Input NOESY feature dimension ({noesy_feat.shape[-1]}) "
                f"does not match expected dimension ({self.noesy_feature_dim})"
            )

        return self.projection(noesy_feat)

if __name__ == '__main__':
    # Example Usage
    batch_size = 2
    seq_len = 50
    noesy_bins = 64 # This is noesy_feature_dim
    pair_dim = 128  # This is token_z_dim
    hidden_dim = 32

    # Test with MLP
    noesy_module_mlp = NOESYModule(
        noesy_feature_dim=noesy_bins,
        token_z_dim=pair_dim,
        noesy_hidden_dim=hidden_dim
    )

    dummy_noesy_data = torch.randn(batch_size, seq_len, seq_len, noesy_bins)
    output_mlp = noesy_module_mlp(dummy_noesy_data)
    print(f"MLP NOESY Module Output Shape: {output_mlp.shape}") # Expected: (2, 50, 50, 128)

    # Test with Linear layer only
    noesy_module_linear = NOESYModule(
        noesy_feature_dim=noesy_bins,
        token_z_dim=pair_dim
    )
    output_linear = noesy_module_linear(dummy_noesy_data)
    print(f"Linear NOESY Module Output Shape: {output_linear.shape}") # Expected: (2, 50, 50, 128)

    # Test dimension mismatch
    try:
        wrong_noesy_data = torch.randn(batch_size, seq_len, seq_len, noesy_bins + 1)
        noesy_module_linear(wrong_noesy_data)
    except ValueError as e:
        print(f"Caught expected error for dim mismatch: {e}")

    assert output_mlp.shape == (batch_size, seq_len, seq_len, pair_dim)
    assert output_linear.shape == (batch_size, seq_len, seq_len, pair_dim)
    print("NOESYModule example usage successful.")
