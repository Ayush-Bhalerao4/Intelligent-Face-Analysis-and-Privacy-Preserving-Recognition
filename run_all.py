"""Runs all three solutions one after another. Figures + CSV tables go to ./outputs/"""
import subprocess, sys, time, os

here = os.path.dirname(os.path.abspath(__file__))
for script in ["solution1_morphing.py", "solution2_clustering.py",
               "solution3_restoration_deid.py"]:
    t = time.time()
    subprocess.run([sys.executable, os.path.join(here, script)], check=True, cwd=here)
    print(f"--- {script} took {time.time() - t:.0f}s ---\n")
print("All done. Open the 'outputs' folder to see every graph.")
