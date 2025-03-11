import torch
from typing import Union, Tuple
import importlib
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from platform import system as platform_system
from tlamacore.utils import Version, _get_dtype
from transformers import __version__ as transformers_version
from triton import __version__ as triton_version
from xformers import __version__ as xformers_version

# Current module version
__version__ = "0.0.1"

# Convert versions to Version objects for comparisons
transformers_version = Version(transformers_version)
xformers_version = Version(xformers_version)

# Detect the operating system
platform_system = platform_system()

# Global variables to control feature availability
HAS_FLASH_ATTENTION = False
HAS_FLASH_ATTENTION_SOFTCAPPING = False
SUPPORTS_BFLOAT16 = False

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

def _is_flash_attention_available(major_version: int):
    """
    Checks if Flash Attention is available and if it supports softcapping.
    
    Args:
        major_version (int): Major CUDA version.
    
    Returns:
        Tuple[bool, bool]: Availability of Flash Attention and softcapping support.
    """
    global HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING
    if major_version >= 8:
        try:
            try:
                from flash_attn.flash_attn_interface import flash_attn_gpu
            except:
                from flash_attn.flash_attn_interface import flash_attn_cuda
            
            HAS_FLASH_ATTENTION = True
            
            from flash_attn import __version__ as flash_attn_version
            HAS_FLASH_ATTENTION_SOFTCAPPING = Version(flash_attn_version) >= Version("2.6.3")
            
            if not HAS_FLASH_ATTENTION_SOFTCAPPING:
                print(
                    "Tlama-Core: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"\
                    "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"\
                    "To update flash-attn, do the below:\n"\
                    '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
                )
        except:
                print(
                    "Tlama-Core: Your Flash Attention 2 installation seems to be broken?\n"\
                    "A possible explanation is you have a new CUDA version which isn't\n"\
                    "yet compatible with FA2? Please file a ticket to tlama-core or FA2.\n"\
                    "We shall now use Xformers instead, which does not have any performance hits!\n"\
                    "We found this negligible impact by benchmarking on 1x A100."
                )

                # Disable Flash Attention
                import transformers.utils.import_utils
                transformers.utils.import_utils.is_flash_attn_2_available = lambda *args, **kwargs: False
                import transformers.utils
                transformers.utils.is_flash_attn_2_available = lambda *args, **kwargs: False

                HAS_FLASH_ATTENTION = False
    else:
        HAS_FLASH_ATTENTION = False
    
    return HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING

def _get_gpu_specs():
    """
    Retrieves GPU specifications and checks if it supports bfloat16.
    
    Returns:
        Tuple: GPU specifications, maximum memory, bfloat16 support, Flash Attention availability.
    """
    global SUPPORTS_BFLOAT16, HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING
    SUPPORTS_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    major_version, minor_version = torch.cuda.get_device_capability()
    
    HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING = _is_flash_attention_available(major_version)
    
    gpu_specs = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_specs.total_memory / 1024 / 1024 / 1024, 3)
    
    return gpu_specs, max_memory, SUPPORTS_BFLOAT16, HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING

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
        raise NotImplementedError("Tlama-Core: Fast inference is not supported with trust_remote_code=True")
    
    if trust_remote_code:
        print("Tlama-Core: WARNING! trust_remote_code=True. Are you sure you want to execute remote code?")
    
    if token is None:
        raise ValueError("Tlama-Core: token is required for loading the model, please provide a token."
                         "You can get a token from Huggingface's model hub."
                         "For more information, please visit: https://huggingface.co/docs/hub/security-tokens")  
        
    gpu_specs, max_memory, SUPPORTS_BFLOAT16, HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING = _get_gpu_specs()
    if dtype is None:
        logger.warning_once("Tlama-Core: dtype is not provided. We will use the default dtype: bfloat16 if available, else float16.")
        dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
    elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
        logger.warning_once("Device does not support bfloat16. Will change to float16.")
        dtype = torch.float16
    
    assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32) 
        
    if device_map is None:
        print("Tlama-Core: WARNING! device_map is not provided. We will use the default device: cuda if available, else cpu.")
        
    from importlib.metadata import version as importlib_version
    try:    vllm_version = f" vLLM: {importlib_version('vllm')}."
    except: vllm_version = ""
    
    statistics = f"""
    ████████╗██╗      █████╗ ███╗   ███╗ █████╗ 
    ╚══██╔══╝██║     ██╔══██╗████╗ ████║██╔══██╗
       ██║   ██║     ███████║██╔████╔██║███████║
       ██║   ██║     ██╔══██║██║╚██╔╝██║██╔══██║
       ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║  ██║
       ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

    Tlama-Core {__version__}: Fast {model_patcher} patching. Transformers: {transformers_version}
    GPU: {gpu_specs.name}. Max memory: {max_memory} GB. Platform: {platform_system}.{vllm_version}
    Torch: {torch.__version__}. CUDA: {torch.version.cuda}. CUDA Toolkit: {gpu_specs.major}.{gpu_specs.minor}. Triton: {triton_version}
    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]
    Free Apache license: https://github.com/eigencore/tlama-core
        """
    print(statistics)

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
        


    