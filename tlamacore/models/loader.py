import torch
from typing import Union, Tuple
import importlib
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from rich.console import Console
from rich.text import Text
from tlamacore.utils import print_first_message
console = Console()



def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    """
    Checks if a package is available and optionally returns its version.
    
    Args:
        pkg_name (str): Name of the package to check.
        return_version (bool): If True, returns the package version.
    
    Returns:
        Union[Tuple[bool, str], bool]: If return_version is True, returns a tuple (availability, version).
                                       Otherwise, returns only the availability.
    """
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    package_exists = False
            else:
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


def init_setup(
    trust_remote_code: bool,
    fast_inference: bool,
    token: str,
    dtype: torch.dtype,
    device_map: str,
    model_patcher: str
):
    """
    Initializes the setup for loading the model.
    
    Args:
        trust_remote_code (bool): Whether to trust remote code.
        fast_inference (bool): Whether to enable fast inference.
        token (str): Authentication token for Huggingface.
        dtype (torch.dtype): Data type to use.
        device_map (str): Device map for model loading.
        model_patcher (str): Name of the model patcher.
    """
    if trust_remote_code and fast_inference:
        console.print("[bold red]Tlama-Core: ERROR! Fast inference is not supported with trust_remote_code=True[/bold red]")
        raise NotImplementedError("Tlama-Core: Fast inference is not supported with trust_remote_code=True")

    if trust_remote_code:
        console.print("[bold yellow]Tlama-Core: WARNING! trust_remote_code=True. Are you sure you want to execute remote code?[/bold yellow]")

    if token is None:
        console.print("[bold red]Tlama-Core: ERROR! token is required for loading the model.[/bold red]")
        raise ValueError("Tlama-Core: token is required for loading the model, please provide a token. "
                         "You can get a token from Huggingface's model hub. "
                         "For more information, please visit: https://huggingface.co/docs/hub/security-tokens")    
        
    print_first_message(device_map=device_map, dtype=dtype, model_patcher=model_patcher)

def load_model_from_hub(
    model_name: str,
    device_map: dict,
    dtype: str,
    token: str,
    max_position_embeddings: int,
    trust_remote_code: bool,
    **kwargs
):
    """
    Loads a model from Huggingface's model hub.
    
    Args:
        model_name (str): Name of the model on Huggingface.
        device_map (dict): Device map for model loading.
        dtype (str): Data type to use.
        token (str): Authentication token for Huggingface.
        max_position_embeddings (int): Maximum number of position embeddings.
        trust_remote_code (bool): Whether to trust remote code.
        **kwargs: Additional arguments for model loading.
    
    Returns:
        model: Model loaded from Huggingface.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map              = device_map,
        torch_dtype             = dtype,
        token                   = token,
        max_position_embeddings = max_position_embeddings,
        trust_remote_code       = trust_remote_code,
        attn_implementation     = "eager",
        **kwargs,
    )
    
    return model

class ModelLoader():
    """
    Class for loading and preparing models for training.
    """
    
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        device_map: dict,
        dtype: str,
        token: str,
        fast_inference: bool,
        max_position_embeddings: int,
        trust_remote_code: bool,
        **kwargs         
                 ):
        
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.device_map = device_map
        self.dtype = dtype
        self.token = token
        self.max_position_embeddings = max_position_embeddings
        self.trust_remote_code = trust_remote_code
        self.kwargs = kwargs
    
    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int,
        fast_inference: bool,
        device_map: dict,
        dtype: str,
        token: str,
        max_position_embeddings: int,
        trust_remote_code: bool,
        model_patcher: str,
        **kwargs
    ):
        """
        Loads a pretrained model from Huggingface.
        
        Args:
            model_name (str): Name of the model on Huggingface.
            max_seq_length (int): Maximum sequence length.
            fast_inference (bool): Whether to enable fast inference.
            device_map (dict): Device map for model loading.
            dtype (str): Data type to use.
            token (str): Authentication token for Huggingface.
            max_position_embeddings (int): Maximum number of position embeddings.
            trust_remote_code (bool): Whether to trust remote code.
            model_patcher (str): Name of the model patcher.
            **kwargs: Additional arguments for model loading.
        """
        setup = init_setup(
            trust_remote_code,
            fast_inference,
            token,
            dtype,
            device_map,
            model_patcher
        )
        


    