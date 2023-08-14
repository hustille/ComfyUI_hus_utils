import pprint
import collections


class DebugExtra:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", )
    RETURN_NAMES = ("extra_pnginfo", "prompt", "state", )
    FUNCTION = "debug_out"
    CATEGORY = "utils/hus"

    def debug_out(self, extra_pnginfo, prompt):

        workflow = extra_pnginfo["workflow"]
        results = {}
        for node in workflow["nodes"]:
            node_id = str(node["id"])
            name = node["type"]
            if "Debug" in name or "Show" in name or "Function" in name or "Evaluate" in name:
                continue

            if "title" in node:
                name += "("+node["title"]+")"
            else:
                if "properties" in node:
                    if "Node name for S&R" in node["properties"]:
                        name += "("+node["properties"]["Node name for S&R"]+")"

            name += "."+node_id

            if "widgets_values" in node and "inputs" not in node:
                results[name] = node["widgets_values"]
            elif node_id in prompt:
                values = prompt[node_id]
                if "inputs" in values:
                    results[name] = {}
                    for widget in values["inputs"].items():
                        (n, v) = widget
                        if type(v) is not str and isinstance(v, collections.abc.Sequence):
                            continue
                        results[name][n] = v
            elif "widgets_values" in node:
                results[name] = node["widgets_values"]
            else:
                results[name] = "no widget values"

        return (pprint.pformat(extra_pnginfo), pprint.pformat(prompt), pprint.pformat(results), )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
