import pandas as pd
import numpy as np
import cv2
from types import SimpleNamespace
from ultralytics.utils import yaml_load
from ultralytics.trackers.bot_sort import BOTSORT
from sort import Sort
import supervision as sv

# ==============================
# Global Variables
# ==============================

row_number = range(0, 8)

y1 = [
    1045, 1010, 975, 940, 905, 870, 835, 800, 765, 730, 695,
    660, 625, 590, 555, 520, 485, 450, 415, 380, 345, 310,
    275, 240, 205, 170, 135, 100, 65, 30, 0
]

y2 = [
    1115, 1150, 1185, 1220, 1255, 1290, 1325, 1360, 1395, 1430,
    1465, 1500, 1535, 1570, 1605, 1640, 1675, 1710, 1745,
    1780, 1815, 1850, 1885, 1920, 1955, 1990, 2025, 2060,
    2095, 2130, 2160
]

# ==============================
# VideoDivision Class
# ==============================

class VideoDivision:
    """
    Divides video frames into specific rows based on video type.
    """

    def video1_division(self, df_video1):
        row1 = df_video1[((df_video1['x'] < 850) & (df_video1['y'] > 1750)) |
                         ((df_video1['x'] < 750) & (df_video1['y'] < 1750))]
        row2 = df_video1[((df_video1['x'] > 750) & (df_video1['x'] < 1150) & (df_video1['y'] < 1750)) |
                         ((df_video1['x'] > 800) & (df_video1['x'] < 1150) & (df_video1['y'] > 1750))]
        row3 = df_video1[(df_video1['x'] > 1200) & (df_video1['x'] < 1600)]
        row4 = df_video1[(df_video1['x'] > 1700) & (df_video1['x'] < 2100)]
        row5 = df_video1[(df_video1['x'] > 2100) & (df_video1['x'] < 2600)]
        row6 = df_video1[((df_video1['x'] > 2600) & (df_video1['x'] < 3050) & (df_video1['y'] > 1750)) |
                         ((df_video1['x'] > 2600) & (df_video1['x'] < 3150) & (df_video1['y'] < 1750))]
        row7_row8 = df_video1[((df_video1['x'] > 3050) & (df_video1['y'] > 1750)) |
                               ((df_video1['x'] > 3150) & (df_video1['y'] <= 1750))]
        row8 = df_video1[((df_video1['x'] > 3350) & (df_video1['y'] >= 1880)) |
                         ((df_video1['x'] > 3450) & (df_video1['y'] <= 1880) & (df_video1['y'] >= 1550)) |
                         ((df_video1['x'] > 3500) & (df_video1['y'] <= 1550) & (df_video1['y'] >= 1100)) |
                         ((df_video1['x'] > 3550) & (df_video1['y'] <= 1100))]
        concat_df = pd.concat([row7_row8, row8])
        row7 = concat_df.drop_duplicates(keep=False)  # Dropping duplicates to get row7
        rows_df_list = [row1, row2, row3, row4, row5, row6, row7, row8]
        return rows_df_list

    def video2_division(self, df_video2):
        row1 = df_video2[((df_video2['x'] < 850) & (df_video2['y'] > 1600)) |
                         ((df_video2['x'] < 750) & (df_video2['y'] < 1600))]
        row2 = df_video2[((df_video2['x'] > 850) & (df_video2['x'] < 1150) & (df_video2['y'] > 1750)) |
                         ((df_video2['x'] < 1080) & (df_video2['x'] > 750) & (df_video2['y'] < 1750))]
        row3 = df_video2[((df_video2['x'] > 1110) & (df_video2['x'] < 1600) & (df_video2['y'] >= 1750)) |
                         ((df_video2['x'] > 1080) & (df_video2['x'] < 1600) & (df_video2['y'] < 1750))]
        row4 = df_video2[(df_video2['x'] > 1700) & (df_video2['x'] < 2100)]
        row5 = df_video2[(df_video2['x'] > 2200) & (df_video2['x'] < 2600)]
        row6 = df_video2[(df_video2['x'] > 2600) & (df_video2['x'] < 3100)]
        row7 = df_video2[((df_video2['x'] > 3100) & (df_video2['x'] < 3330) & (df_video2['y'] > 1750)) |
                         ((df_video2['x'] < 3430) & (df_video2['x'] > 3100) & (df_video2['y'] < 1750))]
        row8 = df_video2[((df_video2['x'] >= 3330) & (df_video2['y'] > 1750)) |
                         ((df_video2['x'] > 3430) & (df_video2['y'] < 1750))]
        rows_df_list = [row1, row2, row3, row4, row5, row6, row7, row8]
        return rows_df_list

    def video3_division(self, df_video3):
        row1 = df_video3[((df_video3['x'] < 800) & (df_video3['y'] > 1750)) |
                         ((df_video3['x'] < 700) & (df_video3['y'] < 1750))]
        row2 = df_video3[((df_video3['x'] < 1150) & (df_video3['y'] > 1750) & (df_video3['x'] > 800)) |
                         ((df_video3['x'] < 1150) & (df_video3['x'] > 700) & (df_video3['y'] < 1750))]
        row3 = df_video3[(df_video3['x'] > 1200) & (df_video3['x'] < 1600)]
        row4 = df_video3[(df_video3['x'] > 1700) & (df_video3['x'] < 2100)]
        row5 = df_video3[(df_video3['x'] > 2100) & (df_video3['x'] < 2600)]
        row6 = df_video3[(df_video3['x'] > 2600) & (df_video3['x'] < 3050)]
        row7 = df_video3[((df_video3['x'] < 3350) & (df_video3['y'] > 1750) & (df_video3['x'] > 3050)) |
                         ((df_video3['x'] < 3450) & (df_video3['x'] > 3100) & (df_video3['y'] <= 1750))]
        row8 = df_video3[((df_video3['x'] > 3350) & (df_video3['y'] > 1750)) |
                         ((df_video3['x'] > 3450) & (df_video3['y'] < 1750))]
        rows_df_list = [row1, row2, row3, row4, row5, row6, row7, row8]
        return rows_df_list

    def video4_division(self, df_video4):
        row1 = df_video4[((df_video4['x'] < 750) & (df_video4['y'] >= 1750)) |
                         ((df_video4['x'] < 650) & (df_video4['y'] < 1750))]
        row2 = df_video4[((df_video4['x'] > 750) & (df_video4['y'] >= 1750) & (df_video4['x'] < 1100)) |
                         ((df_video4['x'] > 650) & (df_video4['y'] < 1750) & (df_video4['x'] < 1100))]
        row3 = df_video4[(df_video4['x'] > 1150) & (df_video4['x'] < 1600)]
        row4 = df_video4[(df_video4['x'] > 1700) & (df_video4['x'] < 2000)]
        row5 = df_video4[(df_video4['x'] > 2200) & (df_video4['x'] < 2500)]
        row6 = df_video4[(df_video4['x'] > 2600) & (df_video4['x'] < 3000)]
        row7 = df_video4[((df_video4['x'] > 3000) & (df_video4['x'] < 3350) & (df_video4['y'] >= 1750)) |
                         ((df_video4['x'] > 3000) & (df_video4['x'] < 3450) & (df_video4['y'] < 1750))]
        row8 = df_video4[((df_video4['x'] > 3350) & (df_video4['y'] >= 1750)) |
                         ((df_video4['x'] > 3450) & (df_video4['y'] < 1750))]
        rows_df_list = [row1, row2, row3, row4, row5, row6, row7, row8]
        return rows_df_list

# Initialize VideoDivision instance
video_division = VideoDivision()

# ==============================
# ByteTrack Class
# ==============================

class ByteTrack:
    """
    Implements the ByteTrack tracking algorithm.
    """

    def __init__(self):
        # Initialize any necessary attributes if needed
        pass

    def return_rowlist(self, df, number):
        """
        Divides the dataframe into specific rows based on the video number.
        """
        if number == 1:
            rows_df = video_division.video1_division(df)
        elif number == 2:
            rows_df = video_division.video2_division(df)
        elif number == 3:
            rows_df = video_division.video3_division(df)
        elif number == 4:
            rows_df = video_division.video4_division(df)
        else:
            raise ValueError("Invalid video number. Must be 1, 2, 3, or 4.")
        return rows_df

    def ret_frame_dets(self, tensor, i, row_number, type):
        """
        Retrieves detections for a specific frame and row.
        """
        xyxy = tensor[i].boxes.xyxy
        conf = tensor[i].boxes.conf
        cxywh = tensor[i].boxes.xywh  # Center x, center y, w, h of bbox

        df1 = pd.DataFrame(xyxy, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        df2 = pd.DataFrame(conf, columns=['conf'])
        df3 = pd.DataFrame(cxywh, columns=['CenterX', 'CenterY', 'w', 'h'])

        df_appended = pd.concat([df1, df2, df3], axis=1)
        final_df = df_appended[df_appended['ymin'] >= 660]  # Detections with ymin >= 660

        rows = self.return_rowlist(final_df, type)
        temp_df = rows[row_number]  # Rows ranging from 0-7 (1-8)
        temp_df = temp_df.drop(['CenterX', 'CenterY', 'w', 'h'], axis=1)
        return temp_df

    def ret_trimmed_df(self, temp_df, y1, y2):
        """
        Trims the dataframe based on y-coordinate bounds.
        """
        # Get detections corresponding to the window specified (y1 <= center_y <= y2)
        trim_df = temp_df[
            (((temp_df['ymin'] + temp_df['ymax']) / 2 >= y1) &
             ((temp_df['ymin'] + temp_df['ymax']) / 2 <= y2))
        ]

        xyxy = trim_df[['xmin', 'ymin', 'xmax', 'ymax']].values
        confidence = trim_df['conf'].values
        return (xyxy, confidence)  # ByteTrack expects a tuple with bbox coordinates and confidence

    def get_bt_count(self, window_listup):
        """
        Counts unique tracked objects using ByteTrack.
        """
        tracker = sv.ByteTrack(minimum_matching_threshold=0.8,
                               track_activation_threshold=0.5,
                               lost_track_buffer=24)
        id_list = []
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

        for frame_ids in track_ids:
            id_list.extend(frame_ids)

        count = len(set(id_list))
        return count

    def count_crops_rows(self, y1_bounds, y2_bounds, row_number, model_resultz, type):
        """
        Counts crops per row within specified bounds using ByteTrack.
        """
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

            print(f'ByteTrack - row{i+1}: {count_in_row}')
            results[f'row{i+1}'] = count_in_row

        return results

    def gen_gt_results(self, version, variant, videotype):
        """
        Generates ground truth results for evaluation.
        
        Parameters:
            version (str): e.g., 'v8', 'v9', 'v10'
            variant (str): e.g., 'v8n', 'v9s', 'v10b'
            videotype (int): 1, 2, 3, or 4
        """
        gt_counts_results = {}
        df_path = f'C:/Users/ashis/Desktop/THESIS/DT_flow/pine_project/data/files/dataframes/{version}/{variant}/video{videotype}_gt.csv'
        df = pd.read_csv(df_path)

        rows = self.return_rowlist(df, videotype)
        gt_counts_results = self.process_gt_counts(rows, y1, y2)

        return gt_counts_results

    def process_gt_counts(self, rows, y1_bounds, y2_bounds):
        """
        Processes ground truth counts.
        (Assuming bt_counter.process_gt_counts is similar to count_crops_rows)
        """
        # Placeholder for actual implementation
        # Replace with actual ground truth processing logic
        gt_counts = {}
        for i, row in enumerate(rows, start=1):
            gt_counts[f'row{i}'] = []  # Populate with actual ground truth data
        return gt_counts

# ==============================
# BoTSORT Class
# ==============================

class BoTSORT:
    """
    Implements the BoT-SORT tracking algorithm.
    """

    def __init__(self, cfg_path):
        """
        Initializes the BoT-SORT tracker with the given configuration path.
        
        Parameters:
            cfg_path (str): Path to the tracker configuration YAML file.
        """
        self.tracker_cfg_path = cfg_path
        cfg_dict = yaml_load(self.tracker_cfg_path)
        self.cfg = SimpleNamespace(**cfg_dict)
        self.tracker = BOTSORT(self.cfg)

    def cust_botsort(self, cust_dets, video):
        """
        Custom implementation of BoT-SORT tracking.
        
        Parameters:
            cust_dets (list): List of DetectionResults per frame.
            video (str): Path to the video file.
        
        Returns:
            int: Count of unique tracked objects.
        """
        video_capture = cv2.VideoCapture(video)
        frame_id = 0
        tracking_ids = []

        while video_capture.isOpened():
            if frame_id >= 60:
                break
            ret, frame = video_capture.read()
            if not ret:
                break

            dets = cust_dets[frame_id]
            tracks = self.tracker.update(dets, frame)

            for track in tracks:
                track_id = int(track[4])
                tracking_ids.append(track_id)

            frame_id += 1

        video_capture.release()
        crop_count = len(set(tracking_ids))
        return crop_count

    class CustomDataFrame(pd.DataFrame):
        """
        Extends pandas DataFrame to include properties for detection data.
        """
        @property
        def _constructor(self):
            return BoTSORT.CustomDataFrame

        @property
        def xywh(self):
            return self[['x', 'y', 'w', 'h']].values.tolist()

        @property
        def conf(self):
            return self['conf'].tolist()

        @property
        def cls(self):
            return self['cls'].tolist()

    def convert_to_df(self, df):
        """
        Converts a pandas DataFrame to a CustomDataFrame format.
        """
        return self.CustomDataFrame(df)

    def get_list_of_dfs(self, results):
        """
        Creates a list of DataFrames for each frame.
        
        Parameters:
            results (list): List of detection results per frame.
        
        Returns:
            list: List of CustomDataFrame instances.
        """
        dataframes_per_frame = []
        for result in results:
            xywh = result.boxes.xywh.numpy()
            conf = result.boxes.conf.numpy()
            cls = result.boxes.cls.numpy()

            data = {
                'x': [item[0] for item in xywh],
                'y': [item[1] for item in xywh],
                'w': [item[2] for item in xywh],
                'h': [item[3] for item in xywh],
                'conf': conf,
                'cls': cls
            }

            df = pd.DataFrame(data)
            custom_df = self.convert_to_df(df)
            dataframes_per_frame.append(custom_df)

        return dataframes_per_frame

    class DetectionResults:
        """
        Structures detection results for compatibility with BoT-SORT.
        """
        def __init__(self, xywh, conf, cls):
            self.xywh = xywh
            self.conf = conf
            self.cls = cls

        @property
        def xywh_list(self):
            return self.xywh.tolist()

        @property
        def conf_list(self):
            return self.conf.tolist()

        @property
        def cls_list(self):
            return self.cls.tolist()

    def get_dets_from_dfs(self, dataframes):
        """
        Converts DataFrames to DetectionResults compatible with BoT-SORT.
        
        Parameters:
            dataframes (list): List of CustomDataFrame instances.
        
        Returns:
            list: List of DetectionResults instances.
        """
        detection_results_per_frame = []
        for df in dataframes:
            xywh = df[['x', 'y', 'w', 'h']].values  # Extract as NumPy array
            conf = df['conf'].values                # Extract as NumPy array
            cls = df['cls'].values                  # Extract as NumPy array

            detection_results = self.DetectionResults(xywh, conf, cls)
            detection_results_per_frame.append(detection_results)

        return detection_results_per_frame

    def count_crops_rows(self, y1_bounds, y2_bounds, row_number, model, videotype, video):
        """
        Counts crops per row using BoT-SORT.
        
        Parameters:
            y1_bounds (list): Lower y-coordinate bounds.
            y2_bounds (list): Upper y-coordinate bounds.
            row_number (list): List of row indices (0-7).
            model (list): Model detection results per frame.
            videotype (int): Video type (1-4).
            video (str): Path to the video file.
        
        Returns:
            dict: Counts of crops per row.
        """
        results = {}

        for i in range(len(row_number)):
            count_in_row = []
            for ymin, ymax in zip(y1_bounds, y2_bounds):
                list_dfs = self.get_list_of_dfs(model)
                cust_dfs_list = []
                for j in range(60):
                    cust_df = self.custom_return_df(list_dfs[j], row_number[i], videotype, ymin, ymax)
                    cust_dfs_list.append(cust_df)
                cust_tens_dets = self.get_dets_from_dfs(cust_dfs_list)
                crops_count = self.cust_botsort(cust_tens_dets, video)
                count_in_row.append(crops_count)
            print(f'BoTSORT - row{i+1}: {count_in_row}')
            results[f'row{i+1}'] = count_in_row

        return results

    def custom_return_df(self, df, row_number, videotype, y1, y2):
        """
        Custom dataframe retrieval for BoT-SORT.
        
        Parameters:
            df (CustomDataFrame): DataFrame for a specific row.
            row_number (int): Row index (0-7).
            videotype (int): Video type (1-4).
            y1 (int): Lower y-coordinate bound.
            y2 (int): Upper y-coordinate bound.
        
        Returns:
            CustomDataFrame: Trimmed DataFrame.
        """
        rows = video_division.return_rowlist(df, videotype)
        temp_df = rows[row_number]
        trim_df = temp_df[(temp_df['y'] > y1) & (temp_df['y'] < y2) & (temp_df['y'] > 660)]
        return trim_df

# ==============================
# SORTTracker Class
# ==============================

class SORTTracker:
    """
    Implements the SORT tracking algorithm.
    """

    def __init__(self):
        """
        Initializes the SORT tracker.
        """
        pass

    def return_rowlist(self, df, type):
        """
        Divides the dataframe into specific rows based on the video type.
        """
        if type == 1:
            video_div_func = video_division.video1_division
        elif type == 2:
            video_div_func = video_division.video2_division
        elif type == 3:
            video_div_func = video_division.video3_division
        elif type == 4:
            video_div_func = video_division.video4_division
        else:
            raise ValueError("Invalid video type. Must be 1, 2, 3, or 4.")
        return video_div_func(df)

    def sort_frame_dets(self, tensor, i, row_number, y1, y2, type):
        """
        Retrieves and processes detections for SORT.
        """
        xyxy = tensor[i].boxes.xyxy
        conf = tensor[i].boxes.conf
        cxywh = tensor[i].boxes.xywh  # Center x, center y, w, h of bbox

        df1 = pd.DataFrame(xyxy, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        df2 = pd.DataFrame(conf, columns=['conf'])
        df3 = pd.DataFrame(cxywh, columns=['CenterX', 'CenterY', 'w', 'h'])

        df_appended = pd.concat([df1, df2, df3], axis=1)
        final_df = df_appended[df_appended['ymin'] >= 660]  # Detections with ymin >= 660

        rows = self.return_rowlist(final_df, type)
        temp_df = rows[row_number].copy()  # Rows ranging from 0-7 (1-8)
        temp_df = temp_df.drop(['CenterX', 'CenterY', 'w', 'h'], axis=1)

        trim_df = temp_df[
            (((temp_df['ymin'] + temp_df['ymax']) / 2 >= y1) &
             ((temp_df['ymin'] + temp_df['ymax']) / 2 <= y2))
        ]

        return trim_df.values

    def process_sort(self, det_results, row_number, y1_lim, y2_lim, type):
        """
        Processes detections using SORT and counts unique objects.
        
        Parameters:
            det_results (list): Detection results per frame.
            row_number (int): Row index (0-7).
            y1_lim (int): Lower y-coordinate bound.
            y2_lim (int): Upper y-coordinate bound.
            type (int): Video type (1-4).
        
        Returns:
            int: Count of unique tracked objects.
        """
        tracker = Sort(max_age=25, min_hits=5, iou_threshold=0.01)
        ids = []

        for i in range(len(det_results)):
            detections = np.empty((0, 5))
            xyxyc = self.sort_frame_dets(det_results, i, row_number, y1_lim, y2_lim, type)
            detections = np.vstack((detections, xyxyc))  # Update detections

            tracked_objects = tracker.update(detections)
            for obj in tracked_objects:
                _, _, _, _, obj_id = map(int, obj)
                ids.append(obj_id)

        crop_count = len(set(ids))
        return crop_count

    def count_crops_rows(self, y1_bounds, y2_bounds, row_number, model_det_results, type):
        """
        Counts crops per row using SORT.
        
        Parameters:
            y1_bounds (list): Lower y-coordinate bounds.
            y2_bounds (list): Upper y-coordinate bounds.
            row_number (list): List of row indices (0-7).
            model_det_results (list): Detection results per frame.
            type (int): Video type (1-4).
        
        Returns:
            dict: Counts of crops per row.
        """
        results = {}

        for i in range(len(row_number)):
            count_in_row = []
            for ymin, ymax in zip(y1_bounds, y2_bounds):
                crops_count = self.process_sort(
                    model_det_results,
                    row_number[i],
                    ymin,
                    ymax,
                    type
                )
                count_in_row.append(crops_count)
            print(f'SORT - row{i+1}: {count_in_row}')
            results[f'row{i+1}'] = count_in_row

        return results
