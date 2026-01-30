
import json
import numpy as np

nb_path = "notebooks/08_train_rail_b.ipynb"

try:
    with open(nb_path, "r") as f:
        nb = json.load(f)
    
    # Look for outputs in cells
    metrics = {}
    for cell in nb["cells"]:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if "text" in output:
                    text = "".join(output["text"])
                    if "FINAL EVALUATION METRICS" in text:
                        # Parse simple text format
                        lines = text.split("\n")
                        for line in lines:
                            if "F1 Micro" in line: metrics["f1_micro"] = line.split(":")[1].strip()
                            if "F1 Macro" in line: metrics["f1_macro"] = line.split(":")[1].strip()
                            if "Hamming Loss" in line: metrics["hamming"] = line.split(":")[1].strip()
    
    print(json.dumps(metrics))

except Exception as e:
    print(f"Error: {e}")
