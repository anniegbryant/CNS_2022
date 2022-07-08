python_to_use <- "/home/osboxes/anaconda3/envs/pyspi/bin/python3"
reticulate::use_python(python_to_use)
library(tidyverse)
library(cowplot)
library(reticulate)
theme_set(theme_cowplot())
source_python("pickle_reader_for_R.py")

pyspi_data <- extract_df_from_pkl("tutorial_example_data/pyspi_calc_table.pkl")