import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from triton import __version__ as triton_version

__version__ = "0.0.1"

def _get_gpu_specs():
    """
    This function is responsible for getting the GPU specs. And know if it supports the required dtype.
    """
    SUPPORTS_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    gpu_specs = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_specs.total_memory / 1024 / 1024 / 1024, 3)
    
    return gpu_specs, max_memory, SUPPORTS_BFLOAT16


def init_setup():
    """
    This function is responsible for initializing the setup for the model.
    """
    
    if trust_remote_code and fast_inference:
        raise NotImplementedError("Tlama-Core: Fast inference is not supported with trust_remote_code=True")
    
    if trust_remote_code:
        print("Tlama-Core: WARNING! trust_remote_code=True. Are you sure you want to trust the remote code?")
    
    if token is None:
        raise ValueError("Tlama-Core: token is required for loading the model, please provide a token."
                         "You can get a token from Huggingface's model hub."
                         "For more information, please visit: https://huggingface.co/docs/hub/security-tokens")  
        
    gpu_specs, max_memory, SUPPORTS_BFLOAT16 = _get_gpu_specs()   
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

        Tlama-Core {__version__}: Fast {model_patcher.__name__[4:-5]} patching. Transformers: {transformers_version}
        GPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform: {platform_system}
        Torch: {torch.__version__}. CUDA: {torch.version.cuda}. CUDA Toolkit: {gpu_stats.major}.{gpu_stats.minor}. Triton: {triton_version}
        Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]
        Free Apache license: http://github.com/unslothai/unsloth
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
    This function is responsible for loading the model from Huggingface's model hub.
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
    This class is responsible for loading the model from Huggingface's model hub.
    Also it is responsible for loading the model and preparing it for training.
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
        device_map: dict,
        dtype: str,
        token: str,
        max_position_embeddings: int,
        trust_remote_code: bool,
        **kwargs
    ):
        
        
        model = load_model_from_hub(
            model_name,
            device_map,
            dtype,
            token,
            max_position_embeddings,
            trust_remote_code,
            **kwargs
        )
    
    
    def set_up_init()