#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#

data_dir = 'data/rock_split'
tabular_dataset_folder = 'tabular_dataset'
output_filename = 'manifest_rocks.csv'

#-----------------------------------------------------------------------------------------#

U.generate_dataset_manifest(data_dir, tabular_dataset_folder, output_filename)

#-----------------------------------------------------------------------------------------#