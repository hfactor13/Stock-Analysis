"""
This script converts the marimo notebook to a self-hosted web assembly HTML file
Marimo and Python wouldn't need to be installed.
"""

from pathlib import Path
import subprocess

html_dir = Path("./html")
html_dir.mkdir(parents = True, exist_ok = True)

cmd = "marimo export html-wasm stock_analysis.py -o ./html/stock_analysis.html --mode edit"
subprocess.run(cmd.split(" "))