# Author: Yijia Xiao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import warnings
import urllib
from pathlib import Path
import torch
import fold


def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v"""
    return not ("esm1v" in model_name)


def load_model_and_alphabet(model_name):
    if model_name.endswith(".pt"):  # treat as filepath
        return load_model_and_alphabet_local(model_name)
    else:
        return load_model_and_alphabet_hub(model_name)


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check if you specified a correct model name?")
    return data


def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data


def load_model_and_alphabet_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    if _has_regression_weights(model_name):
        regression_data = load_regression_hub(model_name)
    else:
        regression_data = None
    return load_model_and_alphabet_core(model_data, regression_data)


def load_model_and_alphabet_local(model_location):
    print('load_model_and_alphabet_local')
    """ Load from local path. The regression weights need to be co-located """
    model_location = Path(model_location)
    model_data = torch.load(str(model_location), map_location="cpu")
    model_name = model_location.stem
    if _has_regression_weights(model_name):
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location="cpu")
    else:
        regression_data = None
    return load_model_and_alphabet_core(model_data, regression_data)


def has_emb_layer_norm_before(model_state):
    """ Determine whether layer norm needs to be applied before the encoder """
    return any(k.startswith("emb_layer_norm_before") for k, param in model_state.items())


def load_model_and_alphabet_core(model_data, regression_data=None):
    print('load_model_and_alphabet_core')
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])
    arch = "msa_transformer"
    alphabet = fold.Alphabet.from_architecture(arch)

    if arch == "msa_transformer":

        # upgrade state dict
        pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
        prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
        prs2 = lambda s: "".join(
            s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
        )
        prs3 = lambda s: s.replace("row", "column") if "row" in s else s.replace("column", "row")
        # prs3 = lambda s: s.replace("column", "row") if "row" in s else s.replace("row", "column")
        # 'max_positions' -> 'max_position_embeddings'
        keys = ['hidden_size', 'embed_positions_msa', 'num_layers', 'hidden_dropout', 'ffn_embed_dim', 'num_attention_heads', 'attention_dropout', 'max_tokens', 'max_position_embeddings']

        # 'activation_dropout': NOT FOUND in Megatron-LM
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items() if pra(arg[0]) in keys}
        # MLP: h -> 4h -> h
        model_args["intermediate_hidden_size"] = model_args["hidden_size"] * 4
        model_args["activation_dropout"] = 0.0

        # model_state = {prs1(prs2(prs3(arg[0]))): arg[1] for arg in model_data["model"].items()}
        model_state = {prs1(prs2(prs3(arg[0]))): arg[1] for arg in model_data["model"]["language_model"].items()}

        if model_args.get("embed_positions_msa", False):
            emb_dim = model_state["msa_position_embedding"].size(-1)
            model_args["embed_positions_msa_dim"] = emb_dim  # initial release, bug: emb_dim==1

        model_type = fold.MegatronMSA

    else:
        raise ValueError("Unknown architecture selected")

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )
    # print(model)
    torch.save(model.state_dict(), 'esm.pt')

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )

    model.load_state_dict(model_state, strict=regression_data is not None)

    return model, alphabet


def megatron_msa_1B():
    """14 layer MSA transformer model with 1B params.

    Returns a tuple of (Model, Alphabet).
    """
    # return load_model_and_alphabet_hub("megatron_msa_1B")
    return load_model_and_alphabet_local("./data/megatron.pt")


def esm1_t34_670M_UR50S():
    """34 layer transformer model with 670M params, trained on Uniref50 Sparse.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t34_670M_UR50S")


def esm1_t34_670M_UR50D():
    """34 layer transformer model with 670M params, trained on Uniref50 Dense.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t34_670M_UR50D")


def esm1_t34_670M_UR100():
    """34 layer transformer model with 670M params, trained on Uniref100.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t34_670M_UR100")


def esm1_t12_85M_UR50S():
    """12 layer transformer model with 85M params, trained on Uniref50 Sparse.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t12_85M_UR50S")


def esm1_t6_43M_UR50S():
    """6 layer transformer model with 43M params, trained on Uniref50 Sparse.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t6_43M_UR50S")


def esm1b_t33_650M_UR50S():
    """33 layer transformer model with 650M params, trained on Uniref50 Sparse.
    This is our best performing model, which will be described in a future publication.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1b_t33_650M_UR50S")


def esm_msa1_t12_100M_UR50S():
    warnings.warn(
        "This model had a minor bug in the positional embeddings, "
        "please use ESM-MSA-1b: fold.pretrained.esm_msa1b_t12_100M_UR50S()",
    )
    return load_model_and_alphabet_hub("esm_msa1_t12_100M_UR50S")


def esm_msa1b_t12_100M_UR50S():
    return load_model_and_alphabet_hub("esm_msa1b_t12_100M_UR50S")


def esm1v_t33_650M_UR90S():
    """33 layer transformer model with 650M params, trained on Uniref90.
    This is model 1 of a 5 model ensemble.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1v_t33_650M_UR90S_1")


def esm1v_t33_650M_UR90S_1():
    """33 layer transformer model with 650M params, trained on Uniref90.
    This is model 1 of a 5 model ensemble.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1v_t33_650M_UR90S_1")


def esm1v_t33_650M_UR90S_2():
    """33 layer transformer model with 650M params, trained on Uniref90.
    This is model 2 of a 5 model ensemble.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1v_t33_650M_UR90S_2")


def esm1v_t33_650M_UR90S_3():
    """33 layer transformer model with 650M params, trained on Uniref90.
    This is model 3 of a 5 model ensemble.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1v_t33_650M_UR90S_3")


def esm1v_t33_650M_UR90S_4():
    """33 layer transformer model with 650M params, trained on Uniref90.
    This is model 4 of a 5 model ensemble.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1v_t33_650M_UR90S_4")


def esm1v_t33_650M_UR90S_5():
    """33 layer transformer model with 650M params, trained on Uniref90.
    This is model 5 of a 5 model ensemble.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1v_t33_650M_UR90S_5")
