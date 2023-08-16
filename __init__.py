from .text import *
from .state import *
from .style import *
from .debug import *

from .sampler.KSamplerWithRefiner import KSamplerWithRefiner

NODE_CLASS_MAPPINGS = {
    "Fetch widget value": FetchNodeValue,
    "3way Prompt Styler": PromptStylerCSV3Way,
    "Text Hash": TextHash,
    "Date Time Format": DateTimeFormat,
    "Batch State": BatchState,
    "Debug Extra": DebugExtra,
    "KSampler With Refiner (Fooocus)": KSamplerWithRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FetchNodeValue": "Fetch Widget Value",
    "PromptStylerCSV3Way": "3way Prompt Styler",
    "TextHash": "Text Hash",
    "DateTimeFormat": "Date Time Format",
    "BatchState": "Batch State",
    "DebugExtra": "Debug Extra",
    "KSamplerWithRefiner" : "KSampler With Refiner (Fooocus)"
}


print("\033[34mhus' utility nodes: \033[92mloaded\033[0m")
