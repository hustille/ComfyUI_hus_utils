import hashlib
import collections
import json


class FetchNodeValue:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "node_name": ("STRING", {"multiline": False}),
                "widget_name": ("STRING", {"multiline": False}),
                "multiple": (["yes", "no"], {"default": "no"}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "get_widget_value"
    CATEGORY = "utils/hus"

    def get_widget_value(self, node_name, widget_name, multiple, extra_pnginfo, prompt):
        workflow = extra_pnginfo["workflow"]
        results = []
        multiple = multiple == "yes"
        for node in workflow["nodes"]:
            node_id = None
            name = node["type"]
            if "properties" in node:
                if "Node name for S&R" in node["properties"]:
                    name = node["properties"]["Node name for S&R"]
            if name == node_name:
                node_id = node["id"]
            else:
                if "title" in node:
                    name = node["title"]
                if name == node_name:
                    node_id = node["id"]

            if node_id is None:
                continue

            values = prompt[str(node_id)]
            if "inputs" in values:
                if widget_name in values["inputs"]:
                    v = values["inputs"][widget_name]
                    if not multiple:
                        return (v, )
                    results.append(v)
                else:
                    raise NameError(f"Widget not found: {node_name}.{widget_name}")
        if not results:
            raise NameError(f"Node not found: {node_name}.{widget_name}")
        return (", ".join(results).strip(", "), )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


class BatchState:
    def __init__(self):
        self.state_hash = False
        self.count = 0

    @staticmethod
    def collect_state(extra_pnginfo, prompt):

        workflow = extra_pnginfo["workflow"]
        results = {}
        if "links" in workflow:
            results["__links"] = workflow["links"]
        for node in workflow["nodes"]:
            node_id = str(node["id"])
            name = node["type"]
            if "Debug" in name or "Show" in name or "Function" in name or "Evaluate" in name:
                continue

            if "widgets_values" in node and "inputs" not in node:
                results[node_id] = node["widgets_values"]
            elif node_id in prompt:
                values = prompt[node_id]
                if "inputs" in values:
                    results[node_id] = {}
                    for widget in values["inputs"].items():
                        (n, v) = widget
                        if type(v) is not str and isinstance(v, collections.abc.Sequence):
                            continue
                        results[node_id][n] = v
            elif "widgets_values" in node:
                results[node_id] = node["widgets_values"]

        result = json.dumps(results, sort_keys=True)
        return hashlib.sha256(result.encode()).hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("NUMBER", "STRING", "NUMBER", "INT", )
    RETURN_NAMES = ("changed", "hash", "count (number)", "count (int)", )
    FUNCTION = "check_state"
    CATEGORY = "utils/hus"

    def check_state(self, extra_pnginfo, prompt):
        old = self.state_hash
        self.state_hash = self.collect_state(extra_pnginfo, prompt)
        if self.state_hash == old:
            self.count += 1
            return (0, self.state_hash, self.count, self.count, )
        self.count = 0
        return (1, self.state_hash, self.count, self.count, )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
