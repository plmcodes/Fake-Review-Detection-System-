"""Shape tests for features and models (tiny random inputs, CPU, a few seconds)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import best_f1_threshold, compute_metrics
from src.features import BEHAVIOUR_COLUMNS, STYLE_COLUMNS, Vocab, behaviour_features, style_features

torch = pytest.importorskip("torch")


def test_style_features_shape_and_values():
    feats = style_features(["I LOVE it!!! My best buy.", "ok"])
    assert feats.shape == (2, len(STYLE_COLUMNS))
    row = feats.iloc[0]
    assert row["exclaim_rate"] > 0 and row["first_person_rate"] > 0 and row["caps_ratio"] > 0
    assert 0 < row["type_token_ratio"] <= 1
    assert not feats.isna().any().any()


def test_vocab_built_and_padded():
    vocab = Vocab.build(["a b c a", "a b d"], max_vocab=10, min_freq=2)
    assert vocab.itos[:3] == ["<pad>", "<unk>", "<bos>"] and set(vocab.itos[3:]) == {"a", "b"}
    ids, lengths = vocab.encode(["a b zzz", "b"], max_len=4)
    assert ids.shape == (2, 4) and lengths.tolist() == [3, 1]
    assert ids[0, 2] == Vocab.UNK and ids[1, 1] == Vocab.PAD


def test_tfidf_fit_on_train_only_has_train_vocabulary():
    from src.config import load_config
    from src.features import make_tfidf

    cfg = load_config(overrides=["features.tfidf.min_df=1"])
    vec = make_tfidf(cfg).fit(["good cheap phone", "bad phone"])
    X = vec.transform(["unseenword phone"])
    assert "unseenword" not in vec.vocabulary_ and X.shape[0] == 1


def test_dense_ae_and_variants_shapes():
    from src.models.ae_dense import AE
    from src.models.ae_variants import VAE, DenoisingAE

    x = torch.randn(8, 384)
    for net in [AE(z=32), DenoisingAE(z=32), VAE(z=32)]:
        net.eval()
        assert net.encode(x).shape == (8, 32)
        assert net.recon_error(x).shape == (8,)
        net.train()
        assert net.loss(x).ndim == 0
    assert VAE(z=16).neg_elbo(x).shape == (8,)


def test_lstm_ae_token_loss_ignores_padding():
    from src.models.lstm_ae import LSTMAutoencoder

    net = LSTMAutoencoder(np.random.randn(50, 16).astype(np.float32), hidden=32, z=8).eval()
    ids = torch.tensor([[5, 6, 7, 0, 0], [8, 9, 0, 0, 0]])
    lengths = torch.tensor([3, 2])
    loss = net.token_loss(ids, lengths)
    assert loss.shape == (2,) and torch.isfinite(loss).all()
    assert net.encode(ids, lengths).shape == (2, 8)


def test_sequence_classifiers_output_one_logit_per_review():
    from src.models.bilstm import BiLSTMClassifier
    from src.models.cnn import TextCNN

    emb = np.random.randn(30, 16).astype(np.float32)
    ids, lengths = torch.randint(1, 30, (4, 7)), torch.tensor([7, 5, 3, 1])
    assert TextCNN(emb, n_filters=4).eval()(ids, lengths).shape == (4,)
    assert BiLSTMClassifier(emb, hidden=8).eval()(ids, lengths).shape == (4,)


def test_hybrid_heads_shapes():
    from src.models.ae_dense import AE
    from src.models.hybrid import JointHybridNet, mlp_head

    assert mlp_head(46)(torch.randn(5, 46)).shape == (5, 1)
    net = JointHybridNet(AE(z=32), n_extra=13, z_dim=32, use_latent=True, use_err=True, hidden=(16,), dropout=0.1).eval()
    logits, err = net(torch.randn(5, 384), torch.randn(5, 13))
    assert logits.shape == (5,) and err.shape == (5,)


def test_behaviour_features_use_no_labels():
    df = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "product_id": ["p", "p", "p"],
                       "rating": [5, 1, 5], "date": ["2014-01-01", "2014-01-01", "2014-02-01"],
                       "label": [1, 0, 0]})
    emb = np.random.randn(3, 8).astype(np.float32)
    a = behaviour_features(df, emb)
    b = behaviour_features(df.assign(label=[0, 1, 1]), emb)   # flipping labels must change nothing
    assert a.shape == (3, len(BEHAVIOUR_COLUMNS)) and a.equals(b)
    assert a.loc[0, "user_max_reviews_per_day"] == 2 and a.loc[2, "max_sim_same_user"] == 0


def test_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    thr = best_f1_threshold(y, s)
    m = compute_metrics(y, s, thr)
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0 and m["tp"] == 3 and m["fp"] == 0
