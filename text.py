import os
import re
import folder_paths
import hashlib
import datetime
import pprint
import collections
import json

class TextHash:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "length": ("INT", {"default": 8, "min": 1, "max": 40, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("hash",)
    FUNCTION = "hash_text"
    CATEGORY = "utils/hus"

    def hash_text(self, text, length):
        return (hashlib.sha256(text.encode()).hexdigest()[:length],)

class DateTimeFormat:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "datetime_fmt"
    CATEGORY = "utils/hus"

    def datetime_fmt(self, format):
        return (datetime.datetime.now().strftime(format), )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
