import os
import re
import folder_paths


class PromptStylerCSV3Way:
    """
    Adapted from "Load Styles CSV" (https://github.com/theUpsider/ComfyUI-Styles_CSV_Loader).
    Split into G and L prompt for SDXL and changed to passthrough for chaining
    """

    @staticmethod
    def load_styles_csv(styles_path: str):
        """Loads csv file with styles. It has only one column.
        Ignore the first row (header).
        positive_prompt are strings separated by comma. Each string is a prompt.
        negative_prompt are strings separated by comma. Each string is a prompt.

        Returns:
            list: List of styles. Each style is a dict with keys: style_name and value: [positive_prompt, negative_prompt]
        """

        styles = {"Error loading styles.csv, check the console": ["",""]}
        if not os.path.exists(styles_path):
            print(f"""Error. No styles.csv found. Put your styles.csv in {styles_path}. Then press "Refresh".
            """)
            return styles
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                styles = [[x.replace('"', '').replace('\n','') for x in re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)] for line in f.readlines()[1:]]
                styles = {x[0]: [x[1],x[2],x[3]] for x in styles}
        except Exception as e:
            print(f"""Error loading styles.csv. Make sure it is in {styles_path}. Then press "Refresh".
                    Error: {e}
            """)
        return styles

    @classmethod
    def INPUT_TYPES(cls):
        cls.styles_folder = os.path.dirname(os.path.realpath(__file__))
        cls.styles_csv = cls.load_styles_csv(os.path.join(cls.styles_folder, "styles.csv"))
        return {
            "required": {
                "styles": (list(cls.styles_csv.keys()),),
                "positive_g": ("STRING", {"multiline": True, "forceInput": True}),
                "positive_l": ("STRING", {"multiline": True, "forceInput": True}),
                "negative": ("STRING", {"multiline": True, "forceInput": True}),
            },

        }

    RETURN_TYPES = ("STRING","STRING","STRING")
    RETURN_NAMES = ("positive prompt (g)", "supporting terms (l)", "negative prompt")
    FUNCTION = "execute"
    CATEGORY = "utils/hus"

    def execute(self, styles, positive_g, positive_l, negative):
        if styles == "none":
            return (positive_g, positive_l, negative)

        pg = self.styles_csv[styles][0]
        if "\u007bprompt\u007d" in pg:
            pg = pg.replace("\u007bprompt\u007d", positive_g)
        else:
            pg += " " + positive_g
        pl = positive_l + ", " + self.styles_csv[styles][1]
        n = negative + ", " + self.styles_csv[styles][2]
        return (pg.strip(" ,"), pl.strip(" ,"), n.strip(" ,"))
