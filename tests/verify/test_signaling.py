from pprint import pprint

from jargon.zoo.signaling import main


def test_signaling() -> None:
    result = main(
        seed=42,
        device="cpu",
        num_elems=50,
        num_attrs=2,
        train_proportion=0.8,
        test_proportion=0.2,
        batch_size=4096,
        vocab_size=50,
        max_len=8,
        sender_input_dim=128,
        receiver_output_dim=128,
        encoder_args={
            "embedding_dim": 8,
            "hidden_sizes": [64],
            "activation_type": "GELU",
            "normalization_type": "LayerNorm",
            "dropout": 0.1,
        },
        sender_args={
            "embedding_dim": 8,
            "hidden_size": 128,
            "num_layers": 2,
            "cell_type": "GRU",
        },
        receiver_args={
            "embedding_dim": 8,
            "hidden_size": 128,
            "num_layers": 2,
            "cell_type": "GRU",
        },
        decoder_args={
            "hidden_sizes": [64],
            "activation_type": "GELU",
            "normalization_type": "LayerNorm",
            "dropout": 0.1,
        },
        lr=1e-3,
        max_epochs=3000,
        show_progress=False,
        use_amp=False,
    )

    train_metrics = result.metrics_fn(
        result.game(result.train_dataset, result.train_dataset)
    )
    test_metrics = result.metrics_fn(
        result.game(result.test_dataset, result.test_dataset)
    )

    pprint(f"train_metrics: {train_metrics}")
    pprint(f"test_metrics: {test_metrics}")

    assert train_metrics["acc_part.mean"] > 0.7
    assert test_metrics["acc_part.mean"] > 0.4
