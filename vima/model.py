from typing import Callable, Literal

import torch
from torch import nn
from torch.nn import Embedding as _Embedding


#utils
class Embedding(_Embedding):
    @property
    def output_dim(self):
        return self.embedding_dim
    
def build_mlp(
    input_dim,
    *,
    hidden_dim,
    output_dim,
    hidden_depth,
    num_layers,
    activation: str | Callable = "relu",
    weight_init: str | Callable = "orthogonal",
    bias_init="zeros",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    add_input_activation: bool | str | Callable = False,
    add_input_norm: bool = False,
    add_output_activation: bool | str | Callable = False,
    add_output_norm: bool = False,
) -> nn.Sequential:
    """
    Tanh is used with orthogonal init => better than relu

    Args:
        norm_type: batchnorm or layernorm applied to intermediate layers
        add_input_activation: add nonlinearty to the input _before_
            procesing a feat from a preceding image encoder => image encoder has a linear layer
            at the end
        add_input_norm: whether to add a norm layr to the input _before_ the mlp computation
        add_output_activation: add nonlinearty to the output _after_ the mlp
        add_output_norm: add norm layer => _after_ mlp comp
    """
    assert (hidden_depth is None) != (num_layers is None), (
        "Either hidden_depth or num_layers must be specified but not both"
        "num_layers is defined as hidden_depth+1"
    )
    if hidden_depth is not None:
        assert hidden_depth >= 0
    if num_layers is not None:
        assert num_layers >= 1
    act_layer = get_activation(activation)

    weight_init = get_initializer(weight_init, activation)
    bias_init = get_initializer(bias_init, activation)

    if norm_type is not None:
        norm_type = norm_type.lower()

    if not norm_type:
        norm_type = nn.Identity
    elif norm_type == "batchnorm":
        norm_type == nn.BatchNorm1d
    elif norm_type == "layernorm":
        norm_type = nn.LayerNorm
    else:
        raise ValueError(f"Unsupposted norm layer: {norm_type}")
    
    hidden_depth = num_layers - 1 if hidden_depth is None else hidden_depth
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), norm_type(hidden_dim), act_layer()]
        for i in range(hidden_depth - 1):
            mods += [
                nn.Linear(hidden_depth, hidden_dim),
                norm_type(hidden_dim),
                act_layer(),
            ]
        mods.append(nn.Linear(hidden_dim, output_dim))
    
    if add_input_norm:
        mods = [norm_type(input_dim)] + mods
    if add_input_activation:
        if add_input_activation is not True:
            act_layer = get_activation(add_input_activation)
        mods = [act_layer()] + mods
    if add_output_norm:
        mods.append(norm_type(output_dim))
    if add_output_activation:
        if add_output_activation is not True:
            act_layer = get_activation(add_output_activation)
        mods.append(act_layer())
    
    for mod in mods:
        if isinstance(mod, nn.Linear):
            weight_init(mod.weight)
            bias_init(mod.bias)
    return nn.Sequential(*mods)


def get_activation(
    activation: str | Callable | None
) -> Callable:
    if not activation:
        return nn.Identity
    elif callable(activation):
        return activation
    ACT_LAYER = {
        "tanh": nn.Tanh,
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        "swish": lambda: nn.SiLU(inplace=True),
        "sigmoid": nn.Sigmoid,
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
    }
    activation = activation.lower()
    assert activation in ACT_LAYER, f"Supported activations: {ACT_LAYER.keys()}"
    return ACT_LAYER[activation]

def get_initializer(
    method: str | Callable,
    activation: str
) -> Callable:
    if isinstance(method, str):
        assert hasattr(
            nn.init, f"{method}_"
        ), f"Initalizer nn.init.{method}_ does not exist"
        if method == "orthogonal":
            try:
                gain = nn.init.calculate_gain(activation)
            except ValueError:
                gain = 1.0
            return lambda x: nn.init.orthogonal_(x, gain=gain)
        else:
            return getattr(nn.init, f"{method}_")
    else:
        assert callable(method)
        return method



class Categorical(torch.distributions.Categorical):
    def mode(self):
        return self.logits.argmax(dim=-1)
    
    

class MultiCategorical(torch.distribution.Distribution):
    def __init__(
            self,
            logits,
            action_dims: list[int]
    ):
        assert logits.dim() >= 2, logits.shape
        
        super().__init__(batch_shape=logits[:-1], validate_args=False)

        self._action_dims = tuple(action_dims)
        assert logits.size(-1) == sum(
            self._action_dims
        ), f"sum of action dims {self._action_dims} != {logits.size(-1)}"
        
        self._dist = [
            Categorical(logits=split)
            for split in torch.split(logits, action_dims, dim=-1)
        ]
    
    def mode(self):
        return torch.stack(
            [torch.argmax(dist.probs, dim=-1) for dist in self._dist], dim=-1
        )
    

    
class ActionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dims: dict[str, int | list[int]],
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01
    ):
        super().__init__()
        
        self._decoders = nn.ModuleDict()
        for k, v, in action_dims.items():
            if isinstance(v, int):
                self._decoders[k] = CategoricalNet(
                    input_dim,
                    action_dim=v,
                    hidden_dim=hidden_dim,
                    hidden_depth=hidden_depth,
                    activation=activation,
                    norm_type=norm_type,
                    last_layer_gain=last_layer_gain
                )
            elif isinstance(v, list):
                self._decoders[k] = MultiCategoricalNet(
                    input_dim,
                    action_dims=v,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    norm_type=norm_type,
                    last_layer_gain=last_layer_gain
                )
            else:
                raise ValueError(f"Invalid action_dims value: {v}")
    
    def forward(self, x: torch.Tensor):
        return {k: v(x) for k, v in self._decoders.items()}

def build_mlp_distribution_net(
    input_dim,
    *,
    output_dim,
    hidden_dim,
    hidden_depth,
    activation: str | Callable = "relu",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    last_layer_gain: float | None = 0.01,
):
    """
    Use orthogonal inti to inti the mlp policy

    Args:
        last_layer_gain: orthogonal init gain for the last fc layer.
        You may want to set ti to a small value to have the gaussian centered around 0.0 in the beginning
    """
    mlp = build_mlp(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        activation=activation,
        weight_init="orthogonal",
        bias_init="zeros",
        norm_type=norm_type
    )
    if last_layer_gain:
        assert last_layer_gain > 0
        nn.init.orthogonal_(mlp[-1].weight, gain=last_layer_gain)
    return mlp


class CategoricalNet(nn.Module):
    def __init__(
        self,
        input_dim,
        *,
        action_dim,
        hidden_dim,
        hidden_depth,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        """
        Use orthogonal initialization to init mlp polict
        
        Args:
            last_layer_gain: orthogonal init gain for the last fc layer
            you want to set it to a small value to make the categorical close to unifrom random at the beginning
        """
        super().__init__()
        self.mlp = build_mlp_distribution_net(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
            last_layer_gain=last_layer_gain,
        )
        self.head = CategoricalHead()
    
    def forward(self, x):
        return self.head(self.mlp(x))
    

class MultiCategoricalNet(nn.Module):
    def __init__(
        self,
        input_dim,
        *,
        action_dims,
        hidden_dim,
        hidden_depth,
        activation:  str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        """
        Orthogonal init to init mlp policy
        splut head does not share nn weights

        Args:
            last_layer_gain: orthogonal init gain for the last fc layer
            you may set it to a small value to make the categorical close to uniform random at the beginning
            set to None to use the default gate
        """

        super().__init__()
        self.mlps = nn.ModuleList()
        for action in action_dims:
            net = build_mlp_distribution_net(
                input_dim=input_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=norm_type,
                last_layer_gain=last_layer_gain
            )
            self.mlps.append(net)
        self.head = MultiCategoricalHead(action_dims)
    
    def forward(self, x):
        return self.head(torch.cat([mlp(x) for mlp in self.mlps], dim=-1))
    


class CategoricalHead(nn.Module):
    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=x)
    
class MultiCategoricalHead(nn.Module):
    def __init__(
        self,
        action_dims: list[int]
    ):
        super().__init__()
        self._action_dims = tuple(action_dims)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> MultiCategorical:
        return MultiCategorical(logits=x, action_dims=self._action_dims)
    

class ActionEmbedding(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        embed_dict: dict[str, nn.Module],
    ):
        super().__init__()
        self.embed_dict = nn.ModuleDict(embed_dict)
        embed_dict_output_dim = sum(
            embed_dict[k].output_dim for k in sorted(embed_dict.keys())
        )

        self._post_layer = (
            nn.Identity()
            if output_dim == embed_dict_output_dim
            else nn.Linear(embed_dict_output_dim, output_dim)
        )

        self._output_dim = output_dim
        self._input_fields_checked = False

    @property
    def output_dim(self):
        return self._output_dim
    
    def forward(
        self,
        x_dict: dict[str, torch.Tensor]
    ):
        if not self._input_fields_checked:
            assert set(x_dict.keys()) == set(self.embed_dict.keys())
            self._input_fields_checked = True
        return self._post_layer(
            torch.cat(
                [self._embed_dict[k] for k in sorted(x_dict.keys())], dim=-1
            )
        )


class ContinuousActionEmbedding(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        input_dim,
        hidden_dim,
        hidden_depth,
    ):
        super().__init__()
        self.layer = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
        )
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor):
        return self._layer(x)
    
