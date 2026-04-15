import subprocess
import sys
import os
import tempfile

# ============================================================
# CONFIGURATION
# ============================================================
SOURCE_EP  = "57618e0a-2c99-45ff-9694-24141b92fa17"   # ORNL source
DEST_EP    = "2bb4c984-3797-11f1-bc5a-02535127e3d7"   # Linux endpoint
OUTPUT_DIR = "/home/fahimcsu/WestUS_IWU_trend/Data_main/rasters/Dayflow/raw"
YEARS      = [
              1986, 1987, 1988, 1989, 1990,
              1991, 1992, 1993, 1994, 1995,
              1996, 1997, 1998, 1999, 2000,
              2001, 2002, 2003, 2004, 2005,
              2006, 2007, 2008, 2009, 2010,
              2011, 2012, 2013, 2014, 2015,
              2016, 2017, 2018, 2019]

# HUC2 prefixes covering all 17 western US states
WESTERN_PREFIX = ("09", "10", "11", "12", "13", "14", "15", "16", "17", "18")

# ============================================================
# HELPER: run a globus CLI command and return output
# ============================================================
def run_globus(args):
    result = subprocess.run(
        ["globus"] + args,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

# ============================================================
# SUBMIT ONE YEAR AT A TIME, WAIT BEFORE MOVING TO NEXT
# ============================================================
for year in YEARS:
    src_path  = (f"/gen101/world-shared/doi-data/OLCF/202312/"
                 f"10.13139_OLCF_2222888/VIC4_RAPID_PRISMAORC2019/{year}/")
    dest_path = f"{OUTPUT_DIR}/{year}/"

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    print(f"\n{'='*50}")
    print(f"[{year}] Listing source files and filtering to western US HUC8s...")

    # List all files in the source directory for this year
    ls_output = run_globus([
        "ls",
        f"{SOURCE_EP}:{src_path}",
        "--format=UNIX"
    ])

    # Filter to western US HUC8s (same logic as Mac/download.py)
    batch_lines = []
    for fname in ls_output.splitlines():
        fname = fname.strip()
        if not fname:
            continue
        try:
            huc8_id = fname.split("_")[2]          # e.g. "14010001C" from "flow_nat_14010001C_1986.nc"
            if huc8_id[:2] in WESTERN_PREFIX:
                batch_lines.append(f"{src_path}{fname} {dest_path}{fname}")
        except IndexError:
            continue

    if not batch_lines:
        print(f"  [!] No western US files found for {year}, skipping.")
        continue

    print(f"  Submitting {len(batch_lines)} files...")

    # Write batch file and submit
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as bf:
        bf.write("\n".join(batch_lines))
        batch_file = bf.name

    try:
        task_id = run_globus([
            "transfer",
            "--batch", batch_file,
            "--label", f"Dayflow_WestUS_{year}",
            "--sync-level", "checksum",
            "--notify", "off",
            SOURCE_EP, DEST_EP,
            "--jmespath", "task_id",
            "--format=UNIX"
        ])
    finally:
        os.unlink(batch_file)

    print(f"  Task ID: {task_id}")

    # Wait for THIS year to finish before moving on
    print(f"  Waiting for {year} transfer to complete...")
    run_globus(["task", "wait", task_id, "--polling-interval", "60"])

    # Check final status
    status = run_globus([
        "task", "show", task_id,
        "--jmespath", "status",
        "--format=UNIX"
    ])
    print(f"  [{year}] Transfer {status}")

    if status != "SUCCEEDED":
        print(f"  ERROR: Transfer for {year} failed! Stopping.")
        sys.exit(1)

print("\nAll years downloaded successfully!")
