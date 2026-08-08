from types import SimpleNamespace
import unittest

import torch
from torch import nn

from minionerec.collaborative import CausalDINEncoder, CollaborativeCausalLM, DINConfig


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.last_inputs_embeds = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, **kwargs):
        self.last_inputs_embeds = inputs_embeds
        return SimpleNamespace(logits=inputs_embeds)


def make_encoder():
    return CausalDINEncoder(
        DINConfig(num_items=6, embedding_dim=4, attention_hidden_dim=8, output_dim=4, dropout=0.0)
    )


class CollaborativeModelTest(unittest.TestCase):
    def test_causal_din_ignores_padding_values(self):
        torch.manual_seed(0)
        encoder = make_encoder().eval()
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        first = encoder(torch.tensor([[1, 2, 6, 6]]), mask)
        second = encoder(torch.tensor([[1, 2, 4, 5]]), mask)
        torch.testing.assert_close(first, second)

    def test_collaborative_vector_replaces_only_placeholder(self):
        torch.manual_seed(0)
        base = FakeCausalLM()
        model = CollaborativeCausalLM(base, make_encoder(), collab_token_id=15)
        input_ids = torch.tensor([[1, 15, 2], [3, 15, 4]])
        histories = torch.tensor([[1, 2, 6], [2, 3, 4]])
        masks = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        original = base.get_input_embeddings()(input_ids).detach()

        model(input_ids, history_item_ids=histories, history_mask=masks)
        actual = base.last_inputs_embeds

        torch.testing.assert_close(actual[:, 0], original[:, 0])
        torch.testing.assert_close(actual[:, 2], original[:, 2])
        self.assertFalse(torch.allclose(actual[:, 1], original[:, 1]))

    def test_requires_exactly_one_placeholder_per_sample(self):
        model = CollaborativeCausalLM(FakeCausalLM(), make_encoder(), collab_token_id=15)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(
                torch.tensor([[1, 2, 3]]),
                history_item_ids=torch.tensor([[1, 2]]),
                history_mask=torch.tensor([[1, 1]], dtype=torch.bool),
            )

    def test_frozen_stage_two_trains_only_projector_by_default(self):
        model = CollaborativeCausalLM(FakeCausalLM(), make_encoder(), collab_token_id=15)
        model.freeze_backbones()
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base_model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.behavior_encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.projector.parameters()))


if __name__ == "__main__":
    unittest.main()
