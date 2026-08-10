import json
import os
import tempfile
import unittest

import numpy as np
import torch

from convert_dataset import load_dataset
from minionerec.indexing import (
    SASRec,
    SASRecConfig,
    evaluate_tokenizer,
    train_sasrec_embeddings,
)
from rq.generate_sidalign_indices import generate
from rq.sidalign_trainer import balanced_codebook_clusters, train_sidalign_tokenizer
from rq.models import SIDAlignRQVAE


class SIDAlignTokenizerTest(unittest.TestCase):
    def test_collaborative_loss_rewards_aligned_pairs(self):
        quantized = torch.eye(4) * 4
        aligned = SIDAlignRQVAE.collaborative_loss(quantized, torch.eye(4))
        shuffled = SIDAlignRQVAE.collaborative_loss(
            quantized, torch.eye(4).roll(1, dims=0)
        )
        self.assertLess(aligned.item(), shuffled.item())

    def test_balanced_codebook_clustering_respects_capacity(self):
        torch.manual_seed(0)
        labels = balanced_codebook_clusters(torch.randn(17, 4), num_clusters=4)
        counts = torch.bincount(labels, minlength=4)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_all_sidalign_losses_backpropagate(self):
        torch.manual_seed(0)
        model = SIDAlignRQVAE(
            in_dim=6,
            num_emb_list=[4, 4],
            e_dim=4,
            layers=[8],
            kmeans_init=False,
            sk_epsilons=[0.0, 0.0],
        )
        labels = [torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1])]
        output = model.forward_sidalign(
            torch.randn(4, 6),
            torch.randn(4, 4),
            labels,
            collaborative_weight=0.01,
            diversity_weight=0.0001,
            use_sk=False,
        )
        output["loss"].backward()
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertGreater(model.encoder.mlp_layers[1].weight.grad.abs().sum().item(), 0)
        for key in (
            "reconstruction_loss",
            "quantization_loss",
            "collaborative_loss",
            "diversity_loss",
        ):
            self.assertTrue(torch.isfinite(output[key]))

    def test_tiny_training_and_index_export(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            content_path = os.path.join(directory, "content.npy")
            cf_path = os.path.join(directory, "cf.npy")
            output_dir = os.path.join(directory, "checkpoint")
            index_path = os.path.join(directory, "items.index.json")
            item_path = os.path.join(directory, "items.json")
            np.save(content_path, rng.normal(size=(8, 6)).astype(np.float32))
            np.save(cf_path, rng.normal(size=(8, 4)).astype(np.float32))
            with open(item_path, "w", encoding="utf-8") as stream:
                json.dump({str(index): {"title": str(index)} for index in range(8)}, stream)

            result = train_sidalign_tokenizer(
                content_path=content_path,
                cf_path=cf_path,
                output_dir=output_dir,
                epochs=1,
                batch_size=8,
                eval_step=1,
                num_emb_list=(4, 4),
                latent_dim=4,
                layers=(8,),
                diversity_clusters=2,
                sk_epsilons=(0.0, 0.003),
                kmeans_iters=2,
                device="cpu",
            )
            generate(result.checkpoint_path, index_path, item_file=item_path, device="cpu")
            with open(index_path, "r", encoding="utf-8") as stream:
                indices = json.load(stream)
            self.assertEqual(len(indices), 8)
            self.assertTrue(all(len(codes) == 2 for codes in indices.values()))
            diagnostics = evaluate_tokenizer(
                index_path, cf_path, neighbor_k=2, sample_size=4
            )
            self.assertEqual(diagnostics["items"], 8)


class SASRecTest(unittest.TestCase):
    def test_converter_loads_flat_preprocessing_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = "Office_Products"
            dataset_dir = os.path.join(directory, dataset)
            os.makedirs(dataset_dir)
            with open(
                os.path.join(dataset_dir, f"{dataset}.item.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump({"0": {"title": "zero"}}, stream)
            with open(
                os.path.join(dataset_dir, f"{dataset}.index.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump({"0": ["<a_0>", "<b_0>"]}, stream)
            header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
            for split in ("train", "valid", "test"):
                with open(
                    os.path.join(dataset_dir, f"{dataset}.{split}.inter"),
                    "w",
                    encoding="utf-8",
                ) as stream:
                    stream.write(header + "u1\t0\t0\n")

            loaded = load_dataset(directory, dataset)
            self.assertEqual(loaded["items"]["0"]["title"], "zero")
            self.assertEqual(set(loaded["splits"]), {"train", "valid", "test"})

    def test_model_scores_all_real_items(self):
        torch.manual_seed(0)
        model = SASRec(
            SASRecConfig(
                num_items=6,
                hidden_dim=4,
                max_length=4,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
            )
        ).eval()
        histories = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        self.assertEqual(model(histories).shape, (2, 4, 4))
        self.assertEqual(model.score_items(histories).shape, (2, 6))
        self.assertEqual(model.export_item_embeddings().shape, (6, 4))

    def test_raw_interactions_export_row_aligned_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            item_path = os.path.join(directory, "items.json")
            train_path = os.path.join(directory, "train.inter")
            valid_path = os.path.join(directory, "valid.inter")
            output_path = os.path.join(directory, "cf.npy")
            with open(item_path, "w", encoding="utf-8") as stream:
                json.dump({str(index): {"title": str(index)} for index in range(8)}, stream)
            header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
            rows = "u1\t0 1 2\t3\nu2\t1 3 4\t5\nu3\t2 4 6\t7\n"
            for path in (train_path, valid_path):
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write(header + rows)
            train_sasrec_embeddings(
                train_file=train_path,
                valid_file=valid_path,
                item_file=item_path,
                output_path=output_path,
                hidden_dim=4,
                max_length=4,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                batch_size=3,
                epochs=1,
                patience=1,
            )
            self.assertEqual(np.load(output_path).shape, (8, 4))
            with open(os.path.splitext(output_path)[0] + ".items.json", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), [str(index) for index in range(8)])


if __name__ == "__main__":
    unittest.main()
