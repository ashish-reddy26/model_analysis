
import os
import numpy as np
import pandas as pd


def adjusting_labels(input_folder, dest_folder):
    
    '''
    A function that takes the folder if text files and creates a new folder with trimmed detections
    input_folder: (str)
    dest_folder1:  (str)
    dest_folder2:  (str)
    
    '''
    
    img_w = 3840
    img_h = 2160
    
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Loop over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            text_file_path = os.path.join(input_folder, filename)
            dest_file_loc = os.path.join(dest_folder, filename)
            
            df = pd.read_csv(text_file_path, header=None, delimiter=' ', names=['class', 'xc', 'yc', 'w', 'h', 'id'])
            
            # -- denormalizing the coordinates and w, h of bbox
            df['xc'] = df['xc'] * img_w
            df['yc'] = df['yc'] * img_h
            
            df['w'] = df['w'] * img_w
            df['h'] = df['h'] * img_h
            
            # -- calculate y1 and trim bboxes that are under 660 pixels to avoid pushbar & delete the 'y1' column
            df['y1'] = df['yc'] + (df['h']) / 2
            df = df[df['y1'] >= 660]
            del df['y1']
            
            # -- normalize them back again
            df['xc'] = np.round(df['xc']/img_w, 6)
            df['yc'] = np.round(df['yc']/img_h, 6)
    
            df['w'] = np.round(df['w']/img_w, 6)
            df['h'] = np.round(df['h']/img_h, 6)
            
            # -- write df to a text file
            df.to_csv(dest_file_loc, sep=' ', index=False, header=False)
    
    print('Processed all files')
