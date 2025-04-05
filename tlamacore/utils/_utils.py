import os
import torch
from rich.console import Console
from packaging.version import Version as TrueVersion
from platform import system as platform_system
import time

from transformers import __version__ as transformers_version
from triton import __version__ as triton_version
from xformers import __version__ as xformers_version


# Ruta al archivo version.txt
version_file = os.path.join(os.path.dirname(__file__), "../../version.txt")

# Leer la versión desde el archivo
with open(version_file, "r") as f:
    __version__ = f.read().strip()

console = Console()




# Detect the operating system
platform_system = platform_system()

# Global variables to control feature availability
HAS_FLASH_ATTENTION = False
HAS_FLASH_ATTENTION_SOFTCAPPING = False
SUPPORTS_BFLOAT16 = False

def get_dtype(dtype):
    pass

def Version(version):
    try:
        return TrueVersion(version)
    except:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Tlama-Core: Could not get version for `{version}`\n"\
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )

def get_gpu_specs():
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


def print_first_message(dtype=None, device_map=None, model_patcher="Trainer"):
    # Convert versions to Version objects for comparisons
    transformers_version_parsed = Version(transformers_version)
    xformers_version_parsed = Version(xformers_version)
    gpu_specs, max_memory, SUPPORTS_BFLOAT16, HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING = get_gpu_specs()
    
    if dtype is None:
        console.print("[bold cyan]Tlama-Core: INFO: dtype is not provided. Using bfloat16 if available, else float16.[/bold cyan]")
        dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
    elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
        console.print("[bold yellow]Device does not support bfloat16. Switching to float16.[/bold yellow]")
        dtype = torch.float16
        
    assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)
    
    if device_map is None:
        console.print("[bold yellow]Tlama-Core: WARNING! device_map is not provided. Using CUDA if available, else CPU.[/bold yellow]") 
        
    from importlib.metadata import version as importlib_version
    try:    vllm_version = f" vLLM: {importlib_version('vllm')}."
    except: vllm_version = ""
    
    # Get current time for display
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create color variables for rich console
    title_color = "grey"
    header_color = "bright_cyan"
    value_color = "bright_green"
    warning_color = "bright_yellow"
    version_color = "bright_blue"
    
    statistics = f"""
    [bold {title_color}]
    ████████╗██╗      █████╗ ███╗   ███╗ █████╗ 
    ╚══██╔══╝██║     ██╔══██╗████╗ ████║██╔══██╗
       ██║   ██║     ███████║██╔████╔██║███████║
       ██║   ██║     ██╔══██║██║╚██╔╝██║██╔══██║
       ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║  ██║
       ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝[/]
        [bold {title_color}]• EigenCore AI Engine •[/]             [italic {version_color}]Started: {current_time}[/]

    [bold {header_color}]╔════════════════════════ ENGINE INFO ════════════════════════╗[/]
    [bold {header_color}]║[/] [bold {title_color}]Tlama-Core[/] [bold {value_color}]{__version__}[/]: Fast {model_patcher} patching
    [bold {header_color}]║[/] [bold {header_color}]Transformers:[/] [bold {value_color}]{transformers_version_parsed}[/]
    [bold {header_color}]╚═════════════════════════════════════════════════════════════╝[/]

    [bold {header_color}]╔════════════════════════ HARDWARE INFO ═══════════════════════╗[/]
    [bold {header_color}]║[/] [bold {header_color}]🖥️  GPU:[/] [bold {value_color}]{gpu_specs.name}[/]
    [bold {header_color}]║[/] [bold {header_color}]🧠 Memory:[/] [bold {value_color}]{max_memory} GB[/]
    [bold {header_color}]║[/] [bold {header_color}]💻 Platform:[/] [bold {value_color}]{platform_system}{vllm_version}[/]
    [bold {header_color}]╚══════════════════════════════════════════════════════════════╝[/]

    [bold {header_color}]╔════════════════════════ SOFTWARE INFO ═══════════════════════╗[/]
    [bold {header_color}]║[/] [bold {header_color}]🔥 Torch:[/] [bold {value_color}]{torch.__version__}[/]
    [bold {header_color}]║[/] [bold {header_color}]⚡ CUDA:[/] [bold {value_color}]{torch.version.cuda}[/] [bold {header_color}]Toolkit:[/] [bold {value_color}]{gpu_specs.major}.{gpu_specs.minor}[/]
    [bold {header_color}]║[/] [bold {header_color}]🚀 Triton:[/] [bold {value_color}]{triton_version}[/]
    [bold {header_color}]║[/] [bold {header_color}]📊 Bfloat16:[/] [bold {"bright_green" if SUPPORTS_BFLOAT16 else warning_color}]{str(SUPPORTS_BFLOAT16).upper()}[/]
    [bold {header_color}]║[/] [bold {header_color}]⚡ FA:[/] [bold {value_color}]Xformers = {xformers_version_parsed}. FA2 = {HAS_FLASH_ATTENTION}[/]
    [bold {header_color}]╚══════════════════════════════════════════════════════════════╝[/]

    [italic {version_color}]© 2025 EigenCore. Free Apache license: https://github.com/eigencore/tlama-core[/]
    """
    console.print(statistics)
    time.sleep(1)