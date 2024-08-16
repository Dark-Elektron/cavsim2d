import subprocess
import os
import sys

# Get current path
sCurPath = os.path.abspath(".")

# Get the command line arguments passed by DAKOTA
num_resp = sys.argv[1]
nodes_only = sys.argv[2]
script_path = sys.argv[3]
paramsfile = sys.argv[4]
resultsfile = sys.argv[5]

# Run the Python script and capture the output
if nodes_only:
    cmd = ["python", os.path.join(script_path, "generate_nodes.py"), paramsfile, resultsfile]
else:
    cmd = ["python", os.path.join(script_path, "py_dakota.py"), paramsfile, resultsfile, "2", "ALL", f"{num_resp }"]

process = subprocess.run(cmd)