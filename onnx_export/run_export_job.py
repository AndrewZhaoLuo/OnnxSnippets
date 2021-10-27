import os
import subprocess
from os import path

if __name__ == "__main__":
    # grab a list of all python files in this dir and run python one by one sequantially
    files_to_run = os.listdir("./onnx_export/")
    files_to_run = [
        f for f in files_to_run if f.startswith("export") and f.endswith(".py")
    ]
    for f in files_to_run:
        print(f"Exporting file: {f}")
        subprocess.run(["python", path.join("onnx_export", f)])
