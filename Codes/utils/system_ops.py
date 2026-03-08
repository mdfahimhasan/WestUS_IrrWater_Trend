# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import shutil
import platform
from glob import glob
from pathlib import Path


def makedirs(directory_list):
    """
    Make directory (if not exists) from a list of directory. Can create multiple directories if provided in the arg as
    a list.

    :rtype: None
    :param directory_list: A list of directories to create.

    :return: None.
    """
    for directory in directory_list:
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)


def clean_and_make_directory(dir_path):
    """
    Removes an existing directory and all it's content. Then, makes a new directory.
    Works for a single directory.

    :param dir_path: Path of the directory.

    :return: None.
    """
    # cleaning the existing directory
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    # making a new directory
    os.makedirs(dir_path, exist_ok=True)


def copy_file(input_dir_or_file, copy_dir, search_by='*.tif', rename=None):
    """
    Copy a file to the specified directory.

    :param input_dir_or_file: File path of input directory/ Path of the file to copy.
    :param copy_dir: File path of copy directory.
    :param search_by: Default set to '*.tif'.
    :param rename: New name of file if required. Default set to None.

    :return: File path of copied file.
    """
    makedirs([copy_dir])
    if '.tif' not in input_dir_or_file:
        input_file = glob(os.path.join(input_dir_or_file, search_by))[0]
        if rename is not None:
            copied_file = os.path.join(copy_dir, f'{rename}.tif')
        else:
            file_name = os.path.basename(input_file)
            copied_file = os.path.join(copy_dir, file_name)

        shutil.copyfile(input_file, copied_file)

    else:
        if rename is not None:
            copied_file = os.path.join(copy_dir, f'{rename}.tif')
        else:
            file_name = os.path.basename(input_dir_or_file)
            copied_file = os.path.join(copy_dir, file_name)

        shutil.copyfile(input_dir_or_file, copied_file)

    return copied_file

# # Need to verify this code across platform before using it in the pipeline. 
# It is currently not used in the pipeline. It is a utility function that can be 
# used in the future if needed.

# def make_gdal_sys_call(gdal_command, args, verbose=True):
#     """
#     Build a GDAL system call list suitable for subprocess.run().
#     Works cross-platform (Windows, macOS, Linux).

#     On Windows, tries to find the GDAL command on PATH first (e.g., via conda).
#     If not found, falls back to the OSGeo4W shell bundled with QGIS.
#     On macOS/Linux, GDAL commands are expected on PATH (via conda or Homebrew).

#     ** followed by code from Sayantan Majumdar.

#     :param gdal_command: GDAL command string, e.g., 'gdal_rasterize'.
#     :param args: List of GDAL command arguments.
#     :param verbose: Set True to print system call info.

#     :return: List[str] — GDAL system call list ready for subprocess.run().

#     :raises FileNotFoundError: If the GDAL command cannot be found on PATH
#         and no fallback is available for the current OS.
#     """
#     gdal_path = shutil.which(gdal_command)

#     if gdal_path is not None:
#         # GDAL command found on PATH (works on any OS — conda, Homebrew, system install)
#         sys_call = [gdal_path] + args

#     else:
#         # GDAL not on PATH — check OS-specific fallback locations
#         fallback_dirs = []

#         if os.name == 'nt':
#             # Windows: check OSGeo4W shell (bundled with QGIS)
#             fallback_dirs = [
#                 Path('C:/Program Files/QGIS 3.22.7/bin'),
#                 Path('C:/OSGeo4W64/bin'),
#             ]

#             # also check if OSGeo4W.bat wrapper is available
#             osgeo4w_bat = Path('C:/Program Files/QGIS 3.22.7/OSGeo4W.bat')
#             if osgeo4w_bat.exists():
#                 sys_call = [str(osgeo4w_bat), gdal_command] + args

#                 if verbose:
#                     print(f'GDAL sys call: {sys_call}')
#                 return sys_call

#         else:
#             # macOS / Linux: check common install locations
#             conda_prefix = os.environ.get('CONDA_PREFIX', '')
#             fallback_dirs = [
#                 Path(conda_prefix) / 'bin' if conda_prefix else None,   # active conda env
#                 Path('/opt/homebrew/bin'),                               # Homebrew (Apple Silicon Mac)
#                 Path('/usr/local/bin'),                                  # Homebrew (Intel Mac) / common Linux
#                 Path('/usr/bin'),                                        # system install (Linux)
#             ]
#             fallback_dirs = [d for d in fallback_dirs if d is not None]

#         # search fallback directories for the GDAL command
#         for directory in fallback_dirs:
#             candidate = directory / gdal_command
#             if candidate.exists():
#                 sys_call = [str(candidate)] + args

#                 if verbose:
#                     print(f'GDAL sys call: {sys_call}')
#                 return sys_call

#         # nothing found — raise error with OS-specific install instructions
#         if os.name == 'nt':
#             raise FileNotFoundError(
#                 f"'{gdal_command}' was not found on PATH or in known fallback locations. "
#                 f"Install GDAL via conda (`conda install gdal`) or update the OSGeo4W/QGIS path."
#             )
#         else:
#             raise FileNotFoundError(
#                 f"'{gdal_command}' was not found on PATH or in common locations "
#                 f"({', '.join(str(d) for d in fallback_dirs)}). "
#                 f"Install GDAL via conda (`conda install gdal`) or Homebrew (`brew install gdal`)."
#             )

#     if verbose:
#         print(f'GDAL sys call: {sys_call}')

#     return sys_call


def assign_cpu_nodes(flags):
    """
    Dynamically assigns CPU nodes based on the operating system and the status of processing flags.

    :param flags: list of bool. Each flag indicates whether a processing step should be skipped (True) or run (False).

    :return: int or None. Number of CPU nodes assigned if processing is required; otherwise, None.
    """
    if not isinstance(flags, (list, tuple)):
        raise TypeError("Flags must be a list or tuple of boolean values.")

    # checking if any flag is False
    if any(not flag for flag in flags):

        # detecting OS
        os_name = platform.system()

        # assigning CPU nodes dynamically based on OS
        # uses os.cpu_count() with a cap to avoid overloading the system
        total_cpus = os.cpu_count() or 4  # fallback to 4 if os.cpu_count() returns None

        if os_name == 'Windows':
            use_cpu_nodes = min(total_cpus - 2, 10)    # local PC — leave 2 cores free, cap at 10
        elif os_name == 'Darwin':
            use_cpu_nodes = min(total_cpus - 3, 10)    # macOS — leave 3 cores free, cap at 10
        elif os_name == 'Linux':
            use_cpu_nodes = min(total_cpus - 4, 40)    # HPC — leave 4 cores free, cap at 40
        else:
            raise ValueError(f'Unsupported OS: {os_name}. Supported: Windows, Darwin (macOS), Linux.')

        print(f'\nUsing {use_cpu_nodes} CPU nodes on {os_name} \n...')
        return use_cpu_nodes

    else:
        return None
