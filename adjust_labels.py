
import os
import numpy as np
import pandas as pd


def adjusting_labels(input_folder, left_folder, right_folder, x_off = 2180):
    
    '''
    A function that takes the folder if text files and creates a new folder with trimmed detections
    input_folder: (str)
    dest_folder1:  (str)
    dest_folder2:  (str)
    
    '''
    
    img_w = 3840
    img_h = 1500
    
    # Ensure the destination folder exists
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)
    
    # Loop over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            
            text_file_path = os.path.join(input_folder, filename)
            left_file_loc = os.path.join(left_folder, filename)
            right_file_loc = os.path.join(right_folder, filename)
            
            df = pd.read_csv(text_file_path, header=None, delimiter=' ', names=['class', 'xc', 'yc', 'w', 'h', 'id'])
            
            # -- denormalizing the center coordinates and w, h of bbox
            df['xc'] = df['xc'] * img_w
            df['yc'] = df['yc'] * img_h
            
            df['w'] = df['w'] * img_w
            df['h'] = df['h'] * img_h
            
            # -- calculate y1 and split the df into 2 halves (left and right)
            
            df['x1'] = df['xc'] - (df['w']) / 2
            
            left_df = df[df['x1'] < x_off]
            right_df = df[df['x1'] >= x_off]
            
            del left_df['x1']
            del right_df['x1']
            
            # -- normalizing left labels
            
            left_df['xc'] = np.round(left_df['xc']/x_off, 6)
            left_df['yc'] = np.round(left_df['yc']/img_h, 6)
    
            left_df['w'] = np.round(left_df['w']/x_off, 6)
            left_df['h'] = np.round(left_df['h']/img_h, 6)
            
            # -- write left_df to a text file
            left_df.to_csv(left_file_loc, sep=' ', index=False, header=False)
            
            # -- normalizing right labels
            
            right_df['xc'] = np.round(right_df['xc']/(img_w-x_off), 6)
            right_df['yc'] = np.round(right_df['yc']/img_h, 6)
    
            right_df['w'] = np.round(right_df['w']/(img_w-x_off), 6)
            right_df['h'] = np.round(right_df['h']/img_h, 6)
            
            right_df.to_csv(right_file_loc, sep=' ', index=False, header=False)
    
    print('Processed all files')
