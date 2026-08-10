# MiniOneRec Collaborative Semantic ID Story

## Problem

MiniOneRec already learns collaborative patterns implicitly from user SID
sequences. Its content-derived SID tree is nevertheless fixed before SFT, so
behaviorally related items may occupy unrelated prefixes. The question is not
whether collaborative information exists, but whether the model should spend
capacity relearning that structure after tokenization.

## Decision

We changed the tokenizer instead of adding a runtime behavior adapter. A SASRec model
trained only on training interactions supplies frozen item CF embeddings.
RQ-VAE still reconstructs item content, while contrastive regularization aligns
its quantized representation with the corresponding CF embedding. Diversity
regularization discourages a small subset of codes from dominating assignment.

This keeps one variable under test: the Semantic ID. MiniOneRec SFT, decoding,
and evaluation are identical between baseline and treatment.

## Learning Value

The project distinguishes information from inductive bias. Both systems observe
the same interactions, but SIDAlign moves their structure into the discrete item
vocabulary. The useful measurements are therefore not only final Recall/NDCG,
but also prefix sharing among CF neighbors, code utilization, convergence, and
head/tail behavior.

An improvement would show that recommendation-aware tokenization reduces the
burden on downstream sequence learning. A neutral result would show that
MiniOneRec's multi-task SFT already absorbs the available collaborative signal.
Either outcome answers a concrete representation question without fabricating
an optimization claim.
