import unittest
import torch
import torch.nn as nn

# Assuming boltz.model.modules.noesy_module is accessible
from boltz.model.modules.noesy_module import NOESYModule

class TestNOESYModule(unittest.TestCase):

    def test_module_creation_linear(self):
        noesy_feat_dim = 64
        token_z_dim = 128
        module = NOESYModule(
            noesy_feature_dim=noesy_feat_dim,
            token_z_dim=token_z_dim,
            noesy_hidden_dim=None # Test linear projection
        )
        self.assertIsInstance(module.projection, nn.Linear)
        self.assertEqual(module.projection.in_features, noesy_feat_dim)
        self.assertEqual(module.projection.out_features, token_z_dim)

    def test_module_creation_mlp(self):
        noesy_feat_dim = 64
        token_z_dim = 128
        hidden_dim = 32
        module = NOESYModule(
            noesy_feature_dim=noesy_feat_dim,
            token_z_dim=token_z_dim,
            noesy_hidden_dim=hidden_dim
        )
        self.assertIsInstance(module.projection, nn.Sequential)
        self.assertEqual(module.projection[0].in_features, noesy_feat_dim)
        self.assertEqual(module.projection[0].out_features, hidden_dim)
        self.assertIsInstance(module.projection[1], nn.ReLU)
        self.assertEqual(module.projection[2].in_features, hidden_dim)
        self.assertEqual(module.projection[2].out_features, token_z_dim)

    def test_forward_pass_linear(self):
        batch_size = 2
        seq_len = 50
        noesy_feat_dim = 64
        token_z_dim = 128

        module = NOESYModule(
            noesy_feature_dim=noesy_feat_dim,
            token_z_dim=token_z_dim,
            noesy_hidden_dim=None
        )

        dummy_noesy_feat = torch.randn(batch_size, seq_len, seq_len, noesy_feat_dim)
        output = module(dummy_noesy_feat)

        self.assertEqual(output.shape, (batch_size, seq_len, seq_len, token_z_dim))

    def test_forward_pass_mlp(self):
        batch_size = 2
        seq_len = 50
        noesy_feat_dim = 64
        token_z_dim = 128
        hidden_dim = 32

        module = NOESYModule(
            noesy_feature_dim=noesy_feat_dim,
            token_z_dim=token_z_dim,
            noesy_hidden_dim=hidden_dim
        )

        dummy_noesy_feat = torch.randn(batch_size, seq_len, seq_len, noesy_feat_dim)
        output = module(dummy_noesy_feat)

        self.assertEqual(output.shape, (batch_size, seq_len, seq_len, token_z_dim))

    def test_forward_pass_dim_mismatch(self):
        noesy_feat_dim = 64
        token_z_dim = 128
        module = NOESYModule(noesy_feature_dim=noesy_feat_dim, token_z_dim=token_z_dim)

        wrong_dim_feat = torch.randn(2, 50, 50, noesy_feat_dim + 1) # Incorrect last dimension
        with self.assertRaisesRegex(ValueError, "Input NOESY feature dimension"):
            module(wrong_dim_feat)

    def test_forward_pass_zero_input_linear(self):
        # Test if zero input produces zero output (before activation, if biases are zero)
        # Linear layers have bias by default. If bias is non-zero, output won't be zero.
        # For this test, we can check if it's consistent.
        noesy_feat_dim = 64
        token_z_dim = 128
        module = NOESYModule(noesy_feature_dim=noesy_feat_dim, token_z_dim=token_z_dim, noesy_hidden_dim=None)

        # Reinitialize weights to be deterministic and bias to zero for this specific test
        with torch.no_grad():
            module.projection.weight.fill_(0.1) # some constant
            if module.projection.bias is not None:
                 module.projection.bias.fill_(0.0)

        dummy_noesy_feat = torch.zeros(2, 10, 10, noesy_feat_dim)
        output = module(dummy_noesy_feat)

        # With zero input and zero bias, output of linear layer should be zero
        self.assertTrue(torch.all(output == 0.0))

    def test_forward_pass_zero_input_mlp(self):
        noesy_feat_dim = 64
        token_z_dim = 128
        hidden_dim = 32
        module = NOESYModule(
            noesy_feature_dim=noesy_feat_dim,
            token_z_dim=token_z_dim,
            noesy_hidden_dim=hidden_dim
        )

        # Reinitialize weights and biases for predictability
        with torch.no_grad():
            module.projection[0].weight.fill_(0.1)
            if module.projection[0].bias is not None:
                module.projection[0].bias.fill_(0.0) # First linear layer bias to 0
            module.projection[2].weight.fill_(0.1)
            if module.projection[2].bias is not None:
                module.projection[2].bias.fill_(0.0) # Second linear layer bias to 0

        dummy_noesy_feat = torch.zeros(2, 10, 10, noesy_feat_dim)
        output = module(dummy_noesy_feat)

        # Input is zero, first linear out is zero, ReLU(0)=0, second linear out is zero.
        self.assertTrue(torch.all(output == 0.0))


if __name__ == '__main__':
    unittest.main()
