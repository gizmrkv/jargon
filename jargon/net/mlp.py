from typing import Any, Dict, List, Tuple, Type

from torch import Tensor, nn


class MLP(nn.Module):
    """Multi-layer perceptron

    Parameters
    ----------
    input_dim : int
        The dimension of the input.
    output_dim : int
        The dimension of the output.
    hidden_sizes : List[int]
        The sizes of the hidden layers.
    activation_type : Type[nn.Module], optional
        The activation function, by default nn.ReLU
    activation_args : Dict[str, Any], optional
        The arguments for the activation function, by default None
    normalization_type : Type[nn.Module], optional
        The normalization layer, by default None
    normalization_args : Dict[str, Any], optional
        The arguments for the normalization layer, by default None
    dropout : float, optional
        The dropout rate, by default 0.0

    Examples
    --------
    >>> import torch
    >>> mlp = MLP(10, 5, [20, 30, 40])
    >>> x = torch.randn(100, 10)
    >>> y, _ = mlp(x)
    >>> y.shape
    torch.Size([100, 5])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation_type: Type[nn.Module] | str = nn.ReLU,
        activation_args: Dict[str, Any] | None = None,
        normalization_type: Type[nn.Module] | str | None = None,
        normalization_args: Dict[str, Any] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes

        if isinstance(activation_type, str):
            activation_type = getattr(nn, activation_type)

        if isinstance(normalization_type, str):
            normalization_type = getattr(nn, normalization_type)

        self.activation_type = activation_type
        self.activation_args = activation_args
        self.normalization_type = normalization_type
        self.normalization_args = normalization_args
        self.dropout = dropout

        dims = [input_dim, *hidden_sizes]
        layers = []
        for i in range(len(dims) - 1):
            layers.extend(
                mini_block(
                    input_dim=dims[i],
                    output_dim=dims[i + 1],
                    activation=activation_type,  # type: ignore
                    activation_args=activation_args,
                    normalization=normalization_type,  # type: ignore
                    normalization_args=normalization_args,
                    dropout=dropout,
                )
            )
        layers.append(nn.Linear(dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        x = self.layers(x)
        return x, None


class MultiDiscreteMLP(nn.Module):
    """Multi-layer perceptron for multi-discrete space

    Apply MLP to a fixed-length tuple of discrete values.
    The input is passed through an Embedding layer and then concatenated.

    Parameters
    ----------
    high : int
        The upper bound of the discrete space.
    n : int
        The number of discrete values.
    output_dim : int
        The dimension of the output.
    embedding_dim : int
        The dimension of the embedding.
    hidden_sizes : List[int]
        The sizes of the hidden layers.
    activation_type : Type[nn.Module], optional
        The activation function, by default nn.ReLU
    activation_args : Dict[str, Any], optional
        The arguments for the activation function, by default None
    normalization_type : Type[nn.Module], optional
        The normalization layer, by default None
    normalization_args : Dict[str, Any], optional
        The arguments for the normalization layer, by default None
    dropout : float, optional
        The dropout rate, by default 0.0

    Examples
    --------
    >>> import torch
    >>> mlp = MultiDiscreteMLP(10, 3, 5, 20, [30, 40])
    >>> x = torch.randint(0, 10, (100, 3))
    >>> y, _ = mlp(x)
    >>> y.shape
    torch.Size([100, 5])
    """

    def __init__(
        self,
        high: int,
        n: int,
        output_dim: int,
        embedding_dim: int,
        hidden_sizes: List[int],
        activation_type: Type[nn.Module] | str = nn.ReLU,
        activation_args: Dict[str, Any] | None = None,
        normalization_type: Type[nn.Module] | str | None = None,
        normalization_args: Dict[str, Any] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.high = high
        self.n = n
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.activation_type = activation_type
        self.activation_args = activation_args
        self.normalization_type = normalization_type
        self.normalization_args = normalization_args
        self.dropout = dropout

        self.embedding = nn.Embedding(high, embedding_dim)
        self.mlp = MLP(
            input_dim=embedding_dim * n,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            activation_type=activation_type,
            activation_args=activation_args,
            normalization_type=normalization_type,
            normalization_args=normalization_args,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        x = self.embedding(x)
        x = x.reshape(x.shape[0], -1)
        x, _ = self.mlp(x)
        return x, None


def mini_block(
    input_dim: int,
    output_dim: int,
    activation: Type[nn.Module] = nn.ReLU,
    activation_args: Dict[str, Any] | None = None,
    normalization: Type[nn.Module] | None = None,
    normalization_args: Dict[str, Any] | None = None,
    dropout: float = 0.0,
) -> List[nn.Module]:
    layers: List[nn.Module] = []
    layers.append(nn.Linear(input_dim, output_dim))

    if normalization is not None:
        normalization_args = normalization_args or {}
        layers.append(normalization(output_dim, **normalization_args))

    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))

    activation_args = activation_args or {}
    layers.append(activation(**activation_args))
    return layers
