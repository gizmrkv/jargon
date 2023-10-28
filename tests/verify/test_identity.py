from jargon.zoo.identity import main


def test_identity() -> None:
    result = main(
        seed=42,
        device="cuda",
        num_elems=50,
        num_attrs=2,
        train_proportion=0.8,
        test_proportion=0.2,
        batch_size=4096,
        model_args={
            "embedding_dim": 8,
            "hidden_sizes": [64, 64],
            "activation_type": "GELU",
        },
        lr=1e-3,
        loss_type="sv",
        max_epochs=500,
        show_progress=True,
        use_amp=True,
    )

    train_metrics = result.metrics_fn(
        result.game(result.train_dataset, result.train_dataset)
    )
    test_metrics = result.metrics_fn(
        result.game(result.test_dataset, result.test_dataset)
    )

    assert train_metrics["acc_part.mean"] > 0.99
    assert test_metrics["acc_part.mean"] > 0.99


def test_identity_pg() -> None:
    result = main(
        seed=42,
        device="cuda",
        num_elems=10,
        num_attrs=2,
        train_proportion=0.8,
        test_proportion=0.2,
        batch_size=4096,
        model_args={
            "embedding_dim": 8,
            "hidden_sizes": [64, 64],
            "activation_type": "GELU",
        },
        lr=1e-3,
        loss_type="pg",
        max_epochs=500,
        show_progress=True,
        use_amp=True,
    )

    train_metrics = result.metrics_fn(
        result.game(result.train_dataset, result.train_dataset)
    )
    test_metrics = result.metrics_fn(
        result.game(result.test_dataset, result.test_dataset)
    )

    assert train_metrics["acc_part.mean"] > 0.85
    assert test_metrics["acc_part.mean"] > 0.75
