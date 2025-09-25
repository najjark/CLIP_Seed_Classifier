import os
import shutil


# output directory for saving plots
def setup_output_dir(output_dir):
    # remove old folder if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # recreate empty folder
    os.makedirs(output_dir)