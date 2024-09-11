import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mot_utilities import det_utilities
from mot_utilities import window_analysis
from deep_sort_realtime.deepsort_tracker import DeepSort
from sort import Sort

class ByteTrack_CropCounter:
    def __init__(self):
        # Initialize constants
        self.horizontal_fov = 118  # degrees
        self.vertical_fov = 69.2  # degrees
        self.width = 3840  # pixels
        self.height = 2160  # pixels
        self.happ = np.radians(self.horizontal_fov) / self.width  # Horizontal angle per pixel in radians
        self.vapp = np.radians(self.vertical_fov) / self.height  # Vertical angle per pixel in radians
        self.y1 = [1045, 1010, 975, 940, 905, 870, 835, 800, 765, 730, 695, 660, 625, 590, 555, 520, 485, 450, 415, 380, 345, 
                   310, 275, 240, 205, 170, 135, 100, 65, 30, 0]
        self.y2 = [1115, 1150, 1185, 1220, 1255, 1290, 1325, 1360, 1395, 1430, 1465, 1500, 1535, 1570, 1605, 1640, 1675, 1710, 
                   1745, 1780, 1815, 1850, 1885, 1920, 1955, 1990, 2025, 2060, 2095, 2130, 2160]

    def calculate_angles(self, xc, yc):
        xcs = np.array(xc)
        ycs = np.array(yc)
        xd = xcs - self.width / 2
        yd = self.height / 2 - ycs
        theta_h = xd * self.happ
        horiz_angle = np.degrees(theta_h)
        return horiz_angle.tolist()
  
    def calculate_fov_for_line(self, fov, dimension, start_pixel, end_pixel, is_horizontal=True):
        degrees_per_pixel = fov / dimension
        line_length_pixels = abs(end_pixel - start_pixel)
        line_fov = line_length_pixels * degrees_per_pixel
        return line_fov

    def calculate_vertical_fov(self):
        vertical_fov = []
        for y1_val, y2_val in zip(self.y1, self.y2):
            fov = self.calculate_fov_for_line(self.vertical_fov, self.height, y1_val, y2_val, is_horizontal=False)
            vertical_fov.append(fov)
        return vertical_fov

    def detect_results(self, model, video_type):
        weight = f"C:/Users/ashis/Desktop/THESIS/DT_flow/det_models/pine_project/data/files/weights/best_{model}.pt"
        video_source = f'C:/Users/ashis/Desktop/THESIS/DT_flow/det_models/pine_project/data/videos/vid{video_type+1}.mp4'
        model = YOLO(weight)
        results = model.predict(source=video_source, conf=.3, verbose=False, save=False)
        return results

    def ret_gt_cropcount(self, df, y1, y2):
        trim_df = df[(df['CenterY'] > y1) & (df['CenterY'] < y2)]
        crop_count = trim_df['id'].nunique()
        return crop_count

    def get_accy(self, gt_count, pred_count):
        error_rate = abs(pred_count - gt_count) / gt_count
        if error_rate > 1:
            return 0
        else:
            accuracy = (1 - error_rate) * 100
            return round(accuracy, 2)

    def process_gt_counts(self, row_data, y1_data, y2_data):
        results = {}
        for row_idx in range(1, len(row_data) + 1):
            row_key = f'gt_row{row_idx}'
            gt_counts = []
            for i in range(len(y1_data)):
                gt_count = self.ret_gt_cropcount(row_data[row_idx - 1], y1_data[i], y2_data[i])
                gt_counts.append(gt_count)
            results[row_key] = gt_counts
        return results

    def calculate_accuracies(self, gt_data, count_data):
        results = {}
        for row_idx in range(1, len(gt_data) + 1):
            accuracies = []
            gt_row = gt_data[f'gt_row{row_idx}']
            count_row = count_data[f'row{row_idx}']
            for i in range(len(self.y1)):
                accuracy = self.get_accy(gt_row[i], count_row[i])
                accuracies.append(accuracy)
            results[f'accuracy{row_idx}'] = accuracies
        return results
    
    def return_rowlist(self, df, type):
        # Dynamically get the function based on type
        func_key = getattr(self, f'video{type}_division')
        # Call the function with df as argument
        return func_key(df)
    
    def gen_gt_results(self, version, variant, videotype):
        '''
        version: v8, v9, v10
        variant: v8n, v9s, v10b
        videotype: 1, 2, 3, 4
        '''
        
        gt_counts_results = {}
        df = pd.read_csv(f'C:/Users/ashis/Desktop/THESIS/DT_flow/det_models/pine_project/data/files/dataframes/{version}/{variant}/video{videotype}_gt.csv')
        
        rows = self.return_rowlist(df, videotype)
        gt_counts_results = self.process_gt_counts(rows, self.y1, self.y2)
        
        return gt_counts_results

    def get_avg_accuracy(self, *accuracy_dicts):
        # Collect all keys from all dictionaries
        keys = set(k for d in accuracy_dicts for k in d.keys())
        
        # Initialize dictionary to store average results
        average_results = {key: [] for key in keys}
        
        # Iterate over each key to calculate average
        for key in keys:
            # Collect all list values corresponding to the current key from all dictionaries
            all_lists = [accuracy_dict.get(key, []) for accuracy_dict in accuracy_dicts if key in accuracy_dict]
            
            # Calculate the average of corresponding values from each list
            if all_lists:
                average_list = [sum(values) / len(values) for values in zip(*all_lists)]
                average_results[key] = average_list
        
        return average_results


    def ret_frame_dets(self, tensor, i, row_number, type):
        
        xyxy = tensor[i].boxes.xyxy
        conf = tensor[i].boxes.conf
        cxywh = tensor[i].boxes.xywh
        df1 = pd.DataFrame(xyxy)
        df2 = pd.DataFrame(conf)
        df3 = pd.DataFrame(cxywh, columns=['CenterX', 'CenterY', 'w', 'h'])
        df_appended = pd.concat([df1, df2, df3], axis=1)
        final_df = df_appended[df_appended.iloc[:, 1] >= 660]
        
        rows = self.return_rowlist(final_df, type)
        temp_df = rows[row_number]  # -- rows - ranging from 0-7 (1-8)
        temp_df.drop(['CenterX', 'CenterY', 'w', 'h'], axis=1, inplace=True)
        
        return temp_df

    def ret_trimmed_df(self, temp_df, y1, y2):

        trim_df = temp_df[((((temp_df.iloc[:, 1] + temp_df.iloc[:, 3]) / 2 >= y1) & 
                             ((temp_df.iloc[:, 1] + temp_df.iloc[:, 3]) / 2 <= y2)))]
        
        xyxy = trim_df.iloc[:, :-1].values
        confidence = trim_df.iloc[:, -1].values
        return (xyxy, confidence)

    def get_bt_count(self, window_listup):
        tracker = sv.ByteTrack(minimum_matching_threshold=.8, track_activation_threshold=.5, lost_track_buffer=24)
        id = []
        count = []
        track_ids = []
        for i in range(60):
            results = window_listup[i]
            boxes, confidences = results
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=np.zeros(len(confidences), dtype=int)
            )
            detections = tracker.update_with_detections(detections)
            ids = detections.tracker_id.tolist()
            track_ids.append(ids)
        for i in range(len(track_ids)):
            for j in range(len(track_ids[i])):
                id.append(track_ids[i][j])
        count = len(set(id))
        return count

    def count_crops_rows(self, y1_bounds, y2_bounds, row_number, model_resultz, type):    
        # -- 'type' - it is the video number that we are working on.
        # -- row_numbe ris list of numbers of rows from 0-7 
        results = {}
        for i in range(len(row_number)):
            count_in_row = []
            for ymin, ymax in zip(y1_bounds, y2_bounds):
                results_window = []
                for j in range(60):
                    df = self.ret_frame_dets(model_resultz, j, row_number[i], type)
                    frame_detections = self.ret_trimmed_df(df, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count = self.get_bt_count(results_window)
                count_in_row.append(crops_count)
            print(f'row{i+1}: {count_in_row}')
            results[f'row{i+1}'] = count_in_row
        return results

    def count_crops_rows_1to6(self, y1_bounds, y2_bounds, xmin_list, xmax_list, model):
        results = {}
        for i in range(len(xmin_list)):
            count_in_row = []
            for ymin, ymax in zip(y1_bounds, y2_bounds):
                results_window = []
                for j in range(60):
                    df = self.ret_frame_dets(model, j)
                    frame_detections = self.ret_trimmed_df(df, xmin_list[i], xmax_list[i], ymin, ymax)
                    results_window.append(frame_detections)
                crops_count = self.get_count(results_window)
                count_in_row.append(crops_count)
            print(f'row{i+1}: {count_in_row}')
            results[f'row{i+1}'] = count_in_row
        return results

    def count_crops_rows_78(self, y1_bounds, y2_bounds, xmin, xmax, limit, model):
        count_in_row = []
        if limit > 20:
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                results_window = []
                for j in range(60):
                    df = self.ret_frame_dets(model, j)
                    frame_detections = self.ret_trimmed_df(df, xmin, xmax, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count1 = self.get_count(results_window)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                results_window = []
                for j in range(60):
                    df = self.ret_frame_dets(model, j)
                    df = df[((df.iloc[:, 2] >= 3050) & (df.iloc[:, 2] <= 3400) & (df.iloc[:, 3] <= 2160) & (df.iloc[:, 3] > 1500)) |
                            ((df.iloc[:, 2] >= 3050) & (df.iloc[:, 2] <= 3500) & (df.iloc[:, 3] <= 1500) & (df.iloc[:, 3] >= 660))]
                    frame_detections = self.ret_trimmed_df(df, xmin, xmax, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count2 = self.get_count(results_window)
                count_in_row.append(crops_count2)
        else:
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                results_window = []
                for j in range(60):
                    df = self.ret_frame_dets(model, j)
                    frame_detections = self.ret_trimmed_df(df, xmin, xmax, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count1 = self.get_count(results_window)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                results_window = []
                for j in range(60):
                    df = self.ret_frame_dets(model, j)
                    df = df[((df.iloc[:, 2] >= 3420) & (df.iloc[:, 3] > 1500)) |
                            ((df.iloc[:, 2] >= 3493) & (df.iloc[:, 3] <= 1500) & (df.iloc[:, 3] >= 660))]
                    frame_detections = self.ret_trimmed_df(df, xmin, xmax, ymin, ymax)
                    results_window.append(frame_detections)
                crops_count2 = self.get_count(results_window)
                count_in_row.append(crops_count2)
        return count_in_row

    def count_crops_for_model(self, model_name, results, xmin_limits, xmax_limits, y1, y2):
        row16 = self.count_crops_rows_1to6(y1, y2, xmin_limits[:6], xmax_limits[:6], results)
        row7 = self.count_crops_rows_78(y1, y2, xmin_limits[6], xmax_limits[6], 25 if model_name == "1" else 21, results)
        row8 = self.count_crops_rows_78(y1, y2, xmin_limits[7], xmax_limits[7], 15, results)
        row16['row7'] = row7
        row16['row8'] = row8
        return row16

    
    def plot_accuracy(accuracy_results_list, model_names, vertical_fov, horz_va1, horz_va2):
        
        threshold = 95  # Set the threshold for accuracy
        fig, axs = plt.subplots(1, 8, figsize=(20, 5))  # Create 1 row and 8 columns of subplots

        # Define specific colors for each model
        colors = ['orange', 'blue', 'cyan', 'magenta', 'green']

        if len(model_names) > len(colors):
            raise ValueError("Not enough unique colors defined for the number of models.")

        for i in range(8):
            for accuracy_results, model_name, color in zip(accuracy_results_list, model_names, colors):
                accuracy_key = f'accuracy{i+1}'
                data_x = vertical_fov
                data_y = accuracy_results[accuracy_key]

                # Plot line segments individually
                for j in range(len(data_x) - 1):
                    segment_color = color
                    axs[i].plot([data_x[j], data_x[j+1]], [data_y[j], data_y[j+1]], marker='o', markersize=.7, color=segment_color, linewidth=0.3, label=model_name if j == 0 else "")

                # Draw a horizontal line at the threshold value
                axs[i].axhline(y=threshold, color='red', linestyle='--', linewidth=.5)

            axs[i].set_title(f'Row{i+1} ({horz_va1[i]}°, {horz_va2[i]}°)', fontsize=10)  # Set the font size here
            axs[i].set_ylim(0, 105)
            axs[i].grid()

        axs[3].set_xlabel('Vertical FOV (degrees°)')      
        axs[0].set_ylabel('Accuracy')  

        # Create a legend for the models and place it above the subplots
        handles = [plt.Line2D([0], [0], color=color, lw=1) for color in colors[:len(model_names)]]
        labels = model_names

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(model_names))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust the top to make room for the legend
        plt.show()
        
    def video1_division(self, df_video1):

        row1 = df_video1[((df_video1['CenterX'] < 850) & (df_video1['CenterY'] > 1750)) | (((df_video1['CenterX'] < 750) & (df_video1['CenterY'] < 1750)))] 
        row2 = df_video1[(df_video1['CenterX'] > 750) & (df_video1['CenterX'] <1150) & (df_video1['CenterY'] < 1750) | ((df_video1['CenterX'] > 800) & (df_video1['CenterX'] < 1150) & (df_video1['CenterY'] > 1750))] 
        row3 = df_video1[(df_video1['CenterX'] > 1200) & (df_video1['CenterX'] < 1600)] 
        row4 = df_video1[(df_video1['CenterX'] > 1700) & (df_video1['CenterX'] < 2100)] 
        row5 = df_video1[(df_video1['CenterX'] > 2100) & (df_video1['CenterX'] < 2600)] 
        row6 = df_video1[((df_video1['CenterX'] > 2600) & (df_video1['CenterX'] < 3050) & (df_video1['CenterY'] > 1750)) | ((df_video1['CenterX'] > 2600) & (df_video1['CenterX'] < 3150) & (df_video1['CenterY'] < 1750))] 

        row7_row8 = df_video1[((df_video1['CenterX'] > 3050) & (df_video1['CenterY'] > 1750)) | ((df_video1['CenterX'] > 3150) & (df_video1['CenterY'] <= 1750))]

        row8 = df_video1[((df_video1['CenterX'] > 3350) & (df_video1['CenterY'] >= 1880)) |
                    ((df_video1['CenterX'] > 3450) & (df_video1['CenterY'] <= 1880) & (df_video1['CenterY'] >= 1550)) |
                    ((df_video1['CenterX'] > 3500) & (df_video1['CenterY'] <= 1550) & (df_video1['CenterY'] >= 1100)) |
                    ((df_video1['CenterX'] > 3550) & (df_video1['CenterY'] <= 1100))]

        concat_df = pd.concat([row7_row8, row8])
        row7 = concat_df.drop_duplicates(keep=False)    # Dropping duplicates (dropping 8th row from rows 7 & 8 - to get row7)
        
        rows_df_list = [row1, row2, row3, row4, row5, row6, row7, row8]
        return rows_df_list

    # video2
    def video2_division(df_video2):
        row1 = df_video2[((df_video2['CenterX'] < 850) & (df_video2['CenterY'] > 1600)) | (((df_video2['CenterX'] < 750) & (df_video2['CenterY'] < 1600)))]
        row2 = df_video2[((df_video2['CenterX'] > 850) & (df_video2['CenterX'] < 1150) & (df_video2['CenterY'] > 1750)) | (((df_video2['CenterX'] < 1080) & (df_video2['CenterX'] > 750) & (df_video2['CenterY'] < 1750)))] 
        row3 = df_video2[((df_video2['CenterX'] > 1110) & (df_video2['CenterX'] < 1600) & (df_video2['CenterY'] >= 1750)) | ((df_video2['CenterX'] > 1080) & (df_video2['CenterX'] < 1600) & (df_video2['CenterY'] < 1750))] 
        row4 = df_video2[(df_video2['CenterX'] > 1700)  & (df_video2['CenterX'] < 2100)]
        row5 = df_video2[(df_video2['CenterX'] > 2200)  & (df_video2['CenterX'] < 2600)]
        row6 = df_video2[(df_video2['CenterX'] > 2600)  & (df_video2['CenterX'] < 3100)]
        row7 = df_video2[((df_video2['CenterX'] > 3100) & (df_video2['CenterX'] < 3330) & (df_video2['CenterY'] > 1750)) | (((df_video2['CenterX'] < 3430) & (df_video2['CenterX'] > 3100) & (df_video2['CenterY'] < 1750)))] 
        row8 = df_video2[((df_video2['CenterX'] >= 3330) & (df_video2['CenterY'] > 1750)) | (((df_video2['CenterX'] > 3430) & (df_video2['CenterY'] < 1750)))]

        rows_df_list = [row1, row2, row3, row4, row5, row6, row7, row8]
        return rows_df_list

    # video 3 

    def video3_division(df_video3):
        
        row1 = df_video3[((df_video3['CenterX'] < 800) & (df_video3['CenterY'] > 1750)) | (((df_video3['CenterX'] < 700) & (df_video3['CenterY'] < 1750)))]
        row2 = df_video3[((df_video3['CenterX'] < 1150) & (df_video3['CenterY'] > 1750) & (df_video3['CenterX'] > 800)) | (((df_video3['CenterX'] < 1150) & (df_video3['CenterX'] > 700) & (df_video3['CenterY'] < 1750)))]
        row3 = df_video3[(df_video3['CenterX'] > 1200)  & (df_video3['CenterX'] < 1600)]
        row4 = df_video3[(df_video3['CenterX'] > 1700)  & (df_video3['CenterX'] < 2100)]
        row5 = df_video3[(df_video3['CenterX'] > 2100)  & (df_video3['CenterX'] < 2600)]
        row6 = df_video3[(df_video3['CenterX'] > 2600)  & (df_video3['CenterX'] < 3050)]
        row7 = df_video3[((df_video3['CenterX'] < 3350) & (df_video3['CenterY'] > 1750) & (df_video3['CenterX'] > 3050)) | (((df_video3['CenterX'] < 3450) & (df_video3['CenterX'] > 3100) & (df_video3['CenterY'] <= 1750)))]
        row8 = df_video3[((df_video3['CenterX'] > 3350) & (df_video3['CenterY'] > 1750)) | (((df_video3['CenterX'] > 3450) & (df_video3['CenterY'] < 1750)))]
        
        rows_df_list = [row1, row2, row3, row4, row5, row6, row7, row8]
        return rows_df_list

    # video 4
    def video4_division(df_video4):
        
        row1 = df_video4[((df_video4['CenterX'] < 750) & (df_video4['CenterY'] >= 1750)) | (((df_video4['CenterX'] < 650) & (df_video4['CenterY'] < 1750)))]
        row2 = df_video4[((df_video4['CenterX'] > 750) & (df_video4['CenterY'] >= 1750) & (df_video4['CenterX'] < 1100)) | ((df_video4['CenterX'] > 650) & (df_video4['CenterY'] < 1750) & (df_video4['CenterX'] < 1100))]
        row3 = df_video4[(df_video4['CenterX'] > 1150)  & (df_video4['CenterX'] < 1600)]
        row4 = df_video4[(df_video4['CenterX'] > 1700)  & (df_video4['CenterX'] < 2000)]
        row5 = df_video4[(df_video4['CenterX'] > 2200)  & (df_video4['CenterX'] < 2500)]
        row6 = df_video4[(df_video4['CenterX'] > 2600)  & (df_video4['CenterX'] < 3000)]
        row7 = df_video4[((df_video4['CenterX'] > 3000) & (df_video4['CenterX'] < 3350) & (df_video4['CenterY'] >=1750)) | ((df_video4['CenterX'] > 3000) & (df_video4['CenterX'] < 3450) & (df_video4['CenterY'] < 1750))]
        row8 = df_video4[((df_video4['CenterX'] > 3350) & (df_video4['CenterY'] >=1750)) | ((df_video4['CenterX'] > 3450) & (df_video4['CenterY'] < 1750))]
        
        rows_df_list = [row1, row2, row3, row4, row5, row6, row7, row8]
        return rows_df_list

# -- customized sort tracker

class SORT_CropCounter:
    
    def __init__(self):
        # Initialize constants
        self.horizontal_fov = 118  # degrees
        self.vertical_fov = 69.2  # degrees
        self.width = 3840  # pixels
        self.height = 2160  # pixels
        self.happ = np.radians(self.horizontal_fov) / self.width  # Horizontal angle per pixel in radians
        self.vapp = np.radians(self.vertical_fov) / self.height  # Vertical angle per pixel in radians
        self.y1 = [1045, 1010, 975, 940, 905, 870, 835, 800, 765, 730, 695, 660, 625, 590, 555, 520, 485, 450, 415, 380, 345, 
                   310, 275, 240, 205, 170, 135, 100, 65, 30, 0]
        self.y2 = [1115, 1150, 1185, 1220, 1255, 1290, 1325, 1360, 1395, 1430, 1465, 1500, 1535, 1570, 1605, 1640, 1675, 1710, 
                   1745, 1780, 1815, 1850, 1885, 1920, 1955, 1990, 2025, 2060, 2095, 2130, 2160]

    def return_detections(self, i, tensor, xmin, xmax, y1, y2):
        xyxy = tensor[i].boxes.xyxy
        conf = tensor[i].boxes.conf
        df1 = pd.DataFrame(xyxy)
        df2 = pd.DataFrame(conf)
        df_appended = pd.concat([df1, df2], axis=1)
        df = df_appended[(df_appended.iloc[:, 1] >= 660) & (df_appended.iloc[:, 4] >= .5)] 
        
        trim_df = df[
            ((df.iloc[:, 0] + df.iloc[:, 2]) / 2 >= xmin) & 
            ((df.iloc[:, 0] + df.iloc[:, 2]) / 2 <= xmax) & 
            ((df.iloc[:, 1] + df.iloc[:, 3]) / 2 >= y1) & 
            ((df.iloc[:, 1] + df.iloc[:, 3]) / 2 <= y2)
        ]  # -- 0-x1, 1-y1, 2-x2, 3-y2
        xyxyc = trim_df.values
        return xyxyc

    def return_detections_row7(self, i, tensor, xmin, xmax, y1, y2):
        xyxy = tensor[i].boxes.xyxy
        conf = tensor[i].boxes.conf
        df1 = pd.DataFrame(xyxy)
        df2 = pd.DataFrame(conf)
        df_appended = pd.concat([df1, df2], axis=1)
        df = df_appended[(df_appended.iloc[:, 1] >= 660) & (df_appended.iloc[:, 4] >= .5)]
        trim_df = df[((df.iloc[:, 2] >= 3050) & (df.iloc[:, 2] <= 3300) & (df.iloc[:, 3] <= 2160) & (df.iloc[:, 3] > 1500)) | 
                     ((df.iloc[:, 2] >= 3050) & (df.iloc[:, 2] <= 3500) & (df.iloc[:, 3] <= 1500) & (df.iloc[:, 3] >= 660))]
        xyxyc = trim_df.values
        return xyxyc

    def process_sort(self, det_results, xmin, xmax, y1_lim, y2_lim):
        tracker = Sort(max_age=25, min_hits=5, iou_threshold=.01)  # -- min_hits=5 found through tuning
        ids = []
        for i in range(29):   # -- len(det_results) - paste this in place of 29 for bigger videos
            detections = np.empty((0, 5))
            xyxyc = self.return_detections(i, det_results, xmin, xmax, y1_lim, y2_lim)
            detections = np.vstack((detections, xyxyc))     # -- updating the detections to the tracker

            tracked_objects = tracker.update(detections)
            for obj in tracked_objects:            
                _, _, _, _, obj_id = map(int, obj)
                ids.append(obj_id)
                
        crop_count = len(set(ids))
        return crop_count

    def process_sort_row7(self, det_results, xmin, xmax, y1_lim, y2_lim):
        tracker = Sort(max_age=25, min_hits=5, iou_threshold=.01)  # -- min_hits=5 found through tuning
        ids = []
        for i in range(29):   # -- len(det_results) - paste this in place of 29 for bigger videos
            detections = np.empty((0, 5))
            xyxyc = self.return_detections_row7(i, det_results, xmin, xmax, y1_lim, y2_lim)
            detections = np.vstack((detections, xyxyc))     # -- updating the detections to the tracker
            tracked_objects = tracker.update(detections)
            for obj in tracked_objects:            
                _, _, _, _, obj_id = map(int, obj)
                ids.append(obj_id)
                
        crop_count = len(set(ids))
        return crop_count
    
    def count_crops_rows_1to6(self, y1_bounds, y2_bounds, xmin_list, xmax_list, model):
        results = {}
        for i in range(len(xmin_list)):
            count_in_row = []   
            for ymin, ymax in zip(y1_bounds, y2_bounds):
                crops_count = self.process_sort(model, xmin_list[i], xmax_list[i], ymin, ymax)
                count_in_row.append(crops_count)
            print(f'row{i+1}: {count_in_row}')
            results[f'row{i+1}'] = count_in_row
        return results
    
    def count_crops_rows_78(self, y1_bounds, y2_bounds, xmin_list, xmax_list, limit, model):
        # -- here we will write a, if - else condition, where 20 is the threshold for the limit 
        # -- for row 7 - limit > 20 as the limit is 25
        # -- for row 8 - limit < 20 as the limit is 15
        if limit > 20:
            # -- logic for row 7
            count_in_row = []   
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                crops_count1 = self.process_sort(model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                crops_count2 = self.process_sort_row7(model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count2)
            return count_in_row
        else:
            # -- logic for row 8
            count_in_row = []   
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                crops_count1 = self.process_sort(model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                crops_count2 = self.process_sort(model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count2)
            return count_in_row
    
    def count_crops_for_model(self, model_name, results, xmin_limits, xmax_limits, y1, y2):
        count_results = self.count_crops_rows_1to6(y1, y2, xmin_limits[:6], xmax_limits[:6], results)
        row7 = self.count_crops_rows_78(y1, y2, xmin_limits[6], xmax_limits[6], 25 if model_name == "1" else 21, results)
        row8 = self.count_crops_rows_78(y1, y2, xmin_limits[7], xmax_limits[7], 15, results)
        count_results['row7'] = row7
        count_results['row8'] = row8
        return count_results


class DeepSORTCounter:
    
    def __init__(self):
        self.horizontal_fov = 118  # degrees
        self.vertical_fov = 69.2  # degrees
        self.width = 3840  # pixels
        self.height = 2160  # pixels
        self.happ = np.radians(self.horizontal_fov) / self.width  # Horizontal angle per pixel in radians
        self.vapp = np.radians(self.vertical_fov) / self.height  # Vertical angle per pixel in radians
        self.y1 = [1045, 1010, 975, 940, 905, 870, 835, 800, 765, 730, 695, 660, 625, 590, 555, 520, 485, 450, 415, 380, 345, 
                   310, 275, 240, 205, 170, 135, 100, 65, 30, 0]
        self.y2 = [1115, 1150, 1185, 1220, 1255, 1290, 1325, 1360, 1395, 1430, 1465, 1500, 1535, 1570, 1605, 1640, 1675, 1710, 
                   1745, 1780, 1815, 1850, 1885, 1920, 1955, 1990, 2025, 2060, 2095, 2130, 2160]

    def extract_detections(self, i, tensor, xmin, xmax, ymin, ymax):
        xywh = tensor[i].boxes.xywh
        conf = tensor[i].boxes.conf
        df1 = pd.DataFrame(xywh, columns=['cx', 'cy', 'w', 'h'])
        df2 = pd.DataFrame(conf, columns=['conf'])
        df_appended = pd.concat([df1, df2], axis=1)
        df_appended['x1'] = df_appended['cx'] - df_appended['w'] / 2
        df_appended['y1'] = df_appended['cy'] - df_appended['h'] / 2
        df_appended = df_appended[['x1', 'y1', 'w', 'h', 'conf']]
        df = df_appended[(df_appended['y1'] >= 660) & (df_appended['conf'] >= 0.5)]
        
        trim_df = df[
            ((df['x1'] + df['w'] / 2) >= xmin) &
            ((df['x1'] + df['w'] / 2) <= xmax) &
            ((df['y1'] + df['h'] / 2) >= ymin) &
            ((df['y1'] + df['h'] / 2) <= ymax)
        ]
        detections = trim_df.apply(lambda row: ([row['x1'], row['y1'], row['w'], row['h']], row['conf']), axis=1).tolist()
        
        return detections

    def extract_detections_row7(self, i, tensor, xmin, xmax, ymin, ymax):
        xywh = tensor[i].boxes.xywh
        conf = tensor[i].boxes.conf
        df1 = pd.DataFrame(xywh, columns=['cx', 'cy', 'w', 'h'])
        df2 = pd.DataFrame(conf, columns=['conf'])
        df_appended = pd.concat([df1, df2], axis=1)
        df_appended['x1'] = df_appended['cx'] - df_appended['w'] / 2
        df_appended['y1'] = df_appended['cy'] - df_appended['h'] / 2
        df_appended = df_appended[['x1', 'y1', 'w', 'h', 'conf']]
        df = df_appended[(df_appended['y1'] >= 660) & (df_appended['conf'] >= 0.5)]
        
        trim_df = df[
            ((df['x1'] >= 3050) & (df['x1'] <= 3300) & (df['y1'] <= 2160) & (df['y1'] > 1500)) | 
            ((df['x1'] >= 3050) & (df['x1'] <= 3500) & (df['y1'] <= 1500) & (df['y1'] >= 660))
        ]
        
        detections = trim_df.apply(lambda row: ([row['x1'], row['y1'], row['w'], row['h']], row['conf']), axis=1).tolist()
        return detections

    def process_deepsort(self, source, det_results, xmin, xmax, y1_lim, y2_lim, row7=False):
        tracker = DeepSort(max_age=25, n_init=3, nms_max_overlap=.5, max_cosine_distance=0.18, max_iou_distance=0.9)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Error: Video source {source} not opened.")
        ids = []
        
        frame_number = 0
        while frame_number < 29:  # Change this condition for videos longer than 1 second
            ret, img = cap.read()
            if not ret:
                break

            if row7:
                detections = self.extract_detections_row7(frame_number, det_results, xmin, xmax, y1_lim, y2_lim)
            else:
                detections = self.extract_detections(frame_number, det_results, xmin, xmax, y1_lim, y2_lim)
            
            tracked_objects = tracker.update_tracks(detections, frame=img)

            for track in tracked_objects:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                obj_id = track.track_id
                ids.append(obj_id)
            frame_number += 1
                
        cap.release()
        crop_count = len(set(ids))
        return crop_count
    
    def count_crops_rows_1to6(self, y1_bounds, y2_bounds, xmin_list, xmax_list, model, source):
        results = {}
        for i in range(len(xmin_list)):
            count_in_row = []   
            for ymin, ymax in zip(y1_bounds, y2_bounds):
                crops_count = self.process_deepsort(source, model, xmin_list[i], xmax_list[i], ymin, ymax)
                count_in_row.append(crops_count)
            print(f'row{i+1}: {count_in_row}')
            results[f'row{i+1}'] = count_in_row
        return results

    def count_crops_rows_78(self, y1_bounds, y2_bounds, xmin_list, xmax_list, limit, model, source):
        if limit > 20:
            count_in_row = []   
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                crops_count1 = self.process_deepsort(source, model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                crops_count2 = self.process_deepsort(source, model, xmin_list, xmax_list, ymin, ymax, row7=True)
                count_in_row.append(crops_count2)
            return count_in_row
        else:
            count_in_row = []   
            for ymin, ymax in zip(y1_bounds[:limit], y2_bounds[:limit]):
                crops_count1 = self.process_deepsort(source, model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count1)
            for ymin, ymax in zip(y1_bounds[limit:], y2_bounds[limit:]):
                crops_count2 = self.process_deepsort(source, model, xmin_list, xmax_list, ymin, ymax)
                count_in_row.append(crops_count2)
            return count_in_row
    
    def count_crops_for_model(self, results, xmin_limits, xmax_limits, y1, y2, source):
        row16 = self.count_crops_rows_1to6(y1, y2, xmin_limits[:6], xmax_limits[:6], results, source)
        row7 = self.count_crops_rows_78(y1, y2, xmin_limits[6], xmax_limits[6], 25, results, source)
        row8 = self.count_crops_rows_78(y1, y2, xmin_limits[7], xmax_limits[7], 15, results, source)
        row16['row7'] = row7
        row16['row8'] = row8
        return row16


def estimate_motion(prev_frame, curr_frame):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(curr_frame, None)

    # Match descriptors using BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate homography using RANSAC to handle outliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

# class MotionCompensatedSort(Sort):
#     def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
#         super().__init__(max_age, min_hits, iou_threshold)
#         self.prev_frame = None

#     def update(self, frame, detections=np.empty((0, 5))):
#         if self.prev_frame is not None:
#             H = estimate_motion(self.prev_frame, frame)
#             if H is not None:
#                 # Apply the homography matrix to adjust the predicted positions of trackers
#                 for trk in self.trackers:
#                     state = trk.predict()
#                     pts = np.array([[state[0], state[1]], [state[2], state[3]]], dtype=np.float32).reshape(-1, 1, 2)
#                     adjusted_pts = cv2.perspectiveTransform(pts, H).reshape(2, 2)
#                     state[0], state[1] = adjusted_pts[0]
#                     state[2], state[3] = adjusted_pts[1]
#                     trk.update(state)

#         self.prev_frame = frame.copy()
#         return super().update(detections)


class MotionCompensatedSort(Sort):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        super().__init__(max_age, min_hits, iou_threshold)
        self.prev_frame = None

    def update(self, frame, detections=np.empty((0, 5))):
        if self.prev_frame is not None:
            H = estimate_motion(self.prev_frame, frame)
            if H is not None:
                # Apply the homography matrix to adjust the predicted positions of trackers
                for trk in self.trackers:
                    state = trk.predict()
                    # print(state)
                    if len(state[0]) >= 4:  # Ensure state has the expected length
                        pts = np.array([[state[0][0], state[0][1]], [state[0][2], state[0][3]]], dtype=np.float32).reshape(-1, 1, 2)
                        adjusted_pts = cv2.perspectiveTransform(pts, H).reshape(2, 2)
                        state[0], state[1] = adjusted_pts[0]
                        state[2], state[3] = adjusted_pts[1]
                        trk.update(state[0])
                    else:
                        print(f"Unexpected state shape: {state}")
                        print(f"Unexpected state shape: {state[0]}")
                        # print(f"Unexpected state shape: {state[1]}")
                        # print(f"Unexpected state shape: {state[2]}")
                        
                        # print(f"Unexpected state shape: {adjusted_pts[0]}")
                        # print(f"Unexpected state shape: {adjusted_pts[1]}")
                        

        self.prev_frame = frame.copy()
        return super().update(detections)
