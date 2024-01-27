
import numpy as np
from utils.param import Param
import threading
import global_storage as gs
import time
import cv2
import config as cf
import matplotlib.pyplot as plt 
from utils.queue_handle import *

#import tflite_runtime.interpreter as tflite
# from primesense import openni2#, nite2
# from primesense import _openni2 as n_api

class laneDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # Init device
        # openni2.initialize(cf.OPENNI_PATH) #
        # self.device = openni2.Device.open_any()
        self.image = 0
        self.param = Param()
        #self.interpreter = tflite.Interpreter(model_path=self.param.lane_segment_model_tflite)
        #self.interpreter.allocate_tensors()
        self.onnx_session = self.param.lane_segment_model_onnx
        self.mask = 0
    def remove_countour_with_area(self, img_thresholded: np.uint8, set_area: int = 1000)-> np.uint8:
            # Tìm các contour trong mask_copy
            contours, hierarchy = cv2.findContours(img_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Chuyển img_thresholded sang ảnh RGB để vẽ contour
            mask_rgb = cv2.cvtColor(img_thresholded, cv2.COLOR_GRAY2RGB)
            
            # Tạo một ảnh trắng với cùng kích thước như img_thresholded
            filled_contour = np.ones_like(mask_rgb) * 255
            counter = 0 
            for i, contour in enumerate(contours):
                # Lấy thông tin về phần tử cha của contour
                parent_idx = hierarchy[0][i][3]
                
                # Nếu không có phần tử cha (parent_idx == -1), vẽ contour
                if parent_idx == -1:
                    
                    area = cv2.contourArea(contour)
                    if area < set_area:
                        # Vẽ contour lên mask_rgb
                        # cv2.drawContours(mask_rgb, [contour], -1, (0, 255, 0), 2)
                        
                        # Vẽ filled contour bằng màu trắng
                        cv2.fillPoly(mask_rgb, [contour], (255, 255, 255))
                    else:
                        counter += 1
                        # Nếu diện tích nhỏ hơn 2200, vẽ filled contour bằng màu đen
                        cv2.fillPoly(filled_contour, [contour], (0, 0, 0))
                else:
                    # Vẽ filled contour bằng màu trắng cho các contour con
                    cv2.fillPoly(filled_contour, [contour], (255, 255, 255))


            new_image = cv2.bitwise_and(mask_rgb, filled_contour)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
            return new_image
        
    def remove_small_dashed_line(self,img_thresholded: np.uint8) -> np.uint8:

        # Lấy kích thước của ảnh img_thresholded
        height, width = img_thresholded.shape[:2]

        # Tính tọa độ của điểm giữa
        center_x = width // 2
        center_y = height // 2 - 50

        # Tạo một mảng NumPy chứa các điểm tạo thành hình tam giác
        triangle_points = np.array([[center_x, center_y], [0, height], [width, height]], np.int32)

        # Chuyển đổi mảng điểm tam giác thành định dạng phù hợp cho hàm fillPoly
        triangle_points = triangle_points.reshape((-1, 1, 2))

        # Tạo một ảnh đen với cùng kích thước như img_thresholded
        mask = np.zeros_like(img_thresholded)

        # Vẽ hình tam giác lên mask
        cv2.fillPoly(mask, [triangle_points], (255, 255, 255))

        # Áp dụng masl để cắt ảnh
        cropped_img = cv2.bitwise_and(img_thresholded, mask)
        
        after_remove_countour_with_area = self.remove_countour_with_area(cropped_img, 1000)
        
        
        
        
        _, binary_cropped_img = cv2.threshold(after_remove_countour_with_area, 1, 255, cv2.THRESH_BINARY)
        

        
        return binary_cropped_img
    
    def deformat_mask(self, mask):
        unique_values, counts = np.unique(mask, return_counts=True)

        if len(unique_values) > 3:
            most_common_values = unique_values[np.argsort(counts)][-3:]
            most_common_values = np.sort(most_common_values)

            mask[~np.isin(mask, most_common_values)] = 0
            mask[mask == most_common_values[0]] = 0
            mask[mask == most_common_values[1]] = 165
            mask[mask == most_common_values[2]] = 0
        elif len(unique_values) == 3:
            mask[mask == unique_values[0]] = 0
            mask[mask == unique_values[1]] = 165
            mask[mask == unique_values[2]] = 0
        elif len(unique_values) == 2:
            mask[mask == unique_values[0]] = 0
            mask[mask == unique_values[1]] = 165
        elif len(unique_values) == 1:

            mask[mask == unique_values[0]] = 0

        return mask
        
    def run(self):

        # input_details = self.interpreter.get_input_details()
        # output_details = self.interpreter.get_output_details()
        while not gs.exit_signal:
        # while True:
        
            image = gs.rgb_seg_frames.get()
            
            # img = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
            # image   = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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

            # The model may require additional preprocessing, such as channel ordering or reshaping.
            # You should check the model's documentation for specific requirements.

            # # Step 3: Make predictions using the loaded ONNX model
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name

            # # Run inference
            input_data = self.image[np.newaxis, ...]  # Add batch dimension
            result = self.onnx_session.run([output_name], {input_name: input_data})

            
            # Extract the predicted mask
            predicted_mask = np.argmax(result[0][0], axis=2)
                    # Display the input image and predicted mask side by side
            predicted_mask = self.deformat_mask(predicted_mask)
           
            predicted_mask_display = np.uint8(predicted_mask)
          
            mask_copy = predicted_mask_display
            mask_copy = cv2.resize(mask_copy, (320, 240))
            
           
            # cv2.imshow('oke', mask_copy)
            # cv2.waitKey(1)
            _, img_thresholded = cv2.threshold(img_thresholded, 1, 255, cv2.THRESH_BINARY_INV)
            
            cropimage = self.remove_small_dashed_line(img_thresholded)
    
            
            final_image = cv2.bitwise_xor(img_thresholded, cropimage)
            _, final_image = cv2.threshold(final_image, 1, 255, cv2.THRESH_BINARY_INV)
            

            # print(np.unique(final_image))
            _, final_image = cv2.threshold(final_image, 1, 255, cv2.THRESH_BINARY)
            
            #convert new_image to binary image
            new_img = cv2.Canny(final_image, 165, 250)
            # new_img = cv2.GaussianBlur(new_img, (7, 7), 0)
            # print("hahahahah")
             
            
            gs.current_img = new_img
            # put_to_queue_no_wait_no_block(self.mask,  gs.mask_img)
            