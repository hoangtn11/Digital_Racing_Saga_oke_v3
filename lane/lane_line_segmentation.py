import onnxruntime
import numpy as np
from utils.param import Param
import threading
import global_storage as gs
import time
import cv2
import config as cf
import matplotlib.pyplot as plt 
from utils.queue_handle import *

class laneDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.image = 0
        self.param = Param()
        self.onnx_session = self.param.lane_segment_model_onnx
        self.mask = 0
        
    def deformat_mask(self, mask):
        unique_values, counts = np.unique(mask, return_counts=True)

        if len(unique_values) > 3:
            most_common_values = unique_values[np.argsort(counts)][-3:]
            most_common_values = np.sort(most_common_values)
            # print(most_common_values)

            mask[~np.isin(mask, most_common_values)] = 0
            mask[mask == most_common_values[0]] = 0
            mask[mask == most_common_values[1]] = 0
            mask[mask == most_common_values[2]] = 255
        elif len(unique_values) == 3:
            # print("mask = 3")
            mask[mask == unique_values[0]] = 0
            mask[mask == unique_values[1]] = 0
            mask[mask == unique_values[2]] = 255
        elif len(unique_values) == 2:
            # print("mask = 2")
            mask[mask == unique_values[0]] = 0
            mask[mask == unique_values[1]] = 0
        elif len(unique_values) == 1:
            # print("mask = 1")
            mask[mask == unique_values[0]] = 0

        return mask

    def deformat_mask_e2e(self, mask):
        unique_values, counts = np.unique(mask, return_counts=True)

        if len(unique_values) > 3:
            most_common_values = unique_values[np.argsort(counts)][-3:]
            most_common_values = np.sort(most_common_values)
            # print(most_common_values)

            mask[~np.isin(mask, most_common_values)] = 0
            mask[mask == most_common_values[0]] = 0
            mask[mask == most_common_values[1]] = 165
            mask[mask == most_common_values[2]] = 255
        elif len(unique_values) == 3:
            # print("mask = 3")
            mask[mask == unique_values[0]] = 0
            mask[mask == unique_values[1]] = 165
            mask[mask == unique_values[2]] = 255
        elif len(unique_values) == 2:
            # print("mask = 2")
            mask[mask == unique_values[0]] = 0
            mask[mask == unique_values[1]] = 165
        elif len(unique_values) == 1:
            # print("mask = 1")
            mask[mask == unique_values[0]] = 0

        return mask
        
    def run(self):
        while not gs.exit_signal:
            image = gs.rgb_seg_frames.get()

            height = image.shape[0]
            two_thirds_height = (3 * height) // 4
            two_thirds_height = height - two_thirds_height
            img_thresholded = image.copy()
            img_thresholded[:two_thirds_height, :] = 0
            
            input_shape = (256, 256)  # Change this to match your model's input shape
            self.image = cv2.resize(img_thresholded, input_shape)
            
            # Load and preprocess the image
            # input_image = input_image.resize(input_shape)  # Resize to match model input size
            self.image = np.array(self.image)  # Convert to NumPy array
            self.image = self.image.astype(np.float32) / 255.0  # Normalize pixel values (assuming 0-255 range)

            # # Step 3: Make predictions using the loaded ONNX model
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name

            # # Run inference
            input_data = self.image[np.newaxis, ...]  # Add batch dimension
            result = self.onnx_session.run([output_name], {input_name: input_data})

           
            # Extract the predicted mask
            predicted_mask_origin = np.argmax(result[0][0], axis=2)
                    # Display the input image and predicted mask side by side
            predicted_mask = self.deformat_mask(predicted_mask_origin.copy())
            predicted_mask_e2e = self.deformat_mask_e2e(predicted_mask_origin.copy())
            
            # Convert predicted_mask to uint8 and scale it to the range [0, 255]
            predicted_mask_display = np.uint8(predicted_mask)
            predicted_mask_display_e2e = np.uint8(predicted_mask_e2e)
            
            # put_to_queue_no_wait_no_block(predicted_mask_display_e2e, gs.e2e_images)
            # put_to_queue_no_wait_no_block(predicted_mask_display_e2e, gs.e2e_images)
            # put_to_queue_no_wait_no_block(predicted_mask_display_e2e, gs.e2e_images)
            put_to_queue_no_wait_no_block(predicted_mask_display, gs.dc_images)

            # put_to_queue_no_wait_no_block(predicted_mask_display_e2e, gs.e2e_images)
            # put_to_queue_no_wait_no_block(predicted_mask_display_e2e, gs.e2e_images)
            # put_to_queue_no_wait_no_block(predicted_mask_display_e2e, gs.e2e_images)
            # put_to_queue_no_wait_no_block(predicted_mask_display_e2e, gs.e2e_images)

            # put_to_queue_no_wait_no_block(predicted_mask_display, gs.dc_images)
           
            mask_copy = predicted_mask_display
            mask_copy = cv2.resize(mask_copy, (320, 240))
                  
            # # Tìm các contour trong mask_copy
            # contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # # Chuyển img_thresholded sang ảnh RGB để vẽ contour
            # mask_rgb = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2RGB)
            
            # # Tạo một ảnh trắng với cùng kích thước như img_thresholded
            # filled_contour = np.ones_like(mask_rgb) * 255
            # counter = 0 
            # for i, contour in enumerate(contours):
            #     # Lấy thông tin về phần tử cha của contour
            #     parent_idx = hierarchy[0][i][3]
                
            #     # Nếu không có phần tử cha (parent_idx == -1), vẽ contour
            #     if parent_idx == -1:
                    
            #         area = cv2.contourArea(contour)
            #         if area >= 1000:
            #             # Vẽ contour lên mask_rgb
            #             # cv2.drawContours(mask_rgb, [contour], -1, (0, 255, 0), 2)
                        
            #             # Vẽ filled contour bằng màu trắng
            #             cv2.fillPoly(mask_rgb, [contour], (255, 255, 255))
            #         else:
            #             counter += 1
            #             # Nếu diện tích nhỏ hơn 2200, vẽ filled contour bằng màu đen
            #             cv2.fillPoly(filled_contour, [contour], (0, 0, 0))
            #     else:
            #         # Vẽ filled contour bằng màu trắng cho các contour con
            #         cv2.fillPoly(filled_contour, [contour], (255, 255, 255))

            # new_image = cv2.bitwise_and(mask_rgb, filled_contour)

            # #convert new_image to binary image
            # new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
            gs.current_img = mask_copy
            # put_to_queue_no_wait_no_block(self.mask,  gs.mask_img)
            
