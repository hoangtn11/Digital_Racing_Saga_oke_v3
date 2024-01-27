import cv2
import numpy as np
import threading
import global_storage as gs
import time
from utils.queue_handle import *

class laneDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.image = 0
  
        self.mask = 0
        
    def enhance_white_color(self, gamma=5):
        # Convert the input image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to enhance contrast
        equ = cv2.equalizeHist(gray)

        # Convert back to color image with enhanced white color
        enhanced_img = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)

        # Apply gamma correction for contrast adjustment
        inv_gamma = 1.0 / gamma
        enhanced_img = np.power(enhanced_img / 255.0, inv_gamma)
        enhanced_img = np.uint8(enhanced_img * 255)
        return enhanced_img
    
    def shadow_remove(self, img):
        rgb_planes = cv2.split(img)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)
        shadowremove = cv2.merge(result_norm_planes)
        return shadowremove
    
    
    

    def draw_image_with_filled_contour(self, mask):
        # Create a copy of the mask
        mask_copy = mask.copy()
        
        # Draw a white line at the bottom of the mask
        cv2.line(mask_copy, (0, mask_copy.shape[0] - 1), (mask_copy.shape[1], mask_copy.shape[0] - 1), (255, 255, 255), 2)
        # Thực hiện flood fill sau khi đã vẽ dòng trắng

        # Tìm các contour trong mask_copy
        contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Chuyển mask_copy sang ảnh RGB để vẽ contour
        mask_rgb = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2RGB)
        # Tạo một ảnh trắng với cùng kích thước như mask_copy
        filled_contour = np.ones_like(mask_rgb) * 255
        
        for i, contour in enumerate(contours):
            # Lấy thông tin về phần tử cha của contour
            parent_idx = hierarchy[0][i][3]
            
            # Nếu không có phần tử cha (parent_idx == -1), vẽ contour
            if parent_idx == -1:
                
                area = cv2.contourArea(contour)
                if area >= 60000:
                    # Vẽ contour lên mask_rgb
                    # cv2.drawContours(mask_rgb, [contour], -1, (0, 255, 0), 2)
                    
                    # Vẽ filled contour bằng màu trắng
                    cv2.fillPoly(mask_rgb, [contour], (255, 255, 255))
                else:
                    # Nếu diện tích nhỏ hơn 2200, vẽ filled contour bằng màu đen
                    cv2.fillPoly(filled_contour, [contour], (0, 0, 0))
            else:
                # Vẽ filled contour bằng màu trắng cho các contour con
                cv2.fillPoly(filled_contour, [contour], (255, 255, 255))
        
        # Tạo ảnh mới bằng phép giao giữa ảnh ban đầu và filled_contour
        new_image = cv2.bitwise_and(mask_rgb, filled_contour)
        #convert new_image to binary image
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
        new_image = self.shadow_remove(new_image)
            
        return new_image

    def apply_canny_filter(self):
        
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
     
        # filter for blue lane lines
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # lọc màu trắng làn đường 
        #(hMin = 2 , sMin = 8, vMin = 96), (hMax = 36 , sMax = 49, vMax = 231)
        lower1, upper1 = np.array([2, 8, 96]), np.array([36, 49, 231])
        #(hMin = 0 , sMin = 0, vMin = 45), (hMax = 127 , sMax = 63, vMax = 162)
        lower2, upper2 = np.array([0, 0, 45]), np.array([127, 63, 162])
        #(hMin = 0 , sMin = 0, vMin = 0), (hMax = 39 , sMax = 255, vMax = 104)
        lower3, upper3 = np.array([0, 0, 0]), np.array([39, 255, 104]) #lọc màu đất .
        #(hMin = 0 , sMin = 0, vMin = 0), (hMax = 45 , sMax = 28, vMax = 105)
        
        mask_1 = cv2.inRange(hsv, lower1, upper1) # dải màu đỏ thứ nhất
        mask_2 = cv2.inRange(hsv, lower2, upper2) # dải màu đỏ thứ hai
        mask_3 = cv2.inRange(hsv, lower3, upper3) # dải màu đỏ thứ ba  
        
        mask_r = cv2.bitwise_or(mask_1, mask_2) # kết hợp 2 kết quả từ 2 dải màu khác nhau
        mask_r = cv2.subtract(mask_r, mask_3) # kết hợp kết quả từ 3 dải màu khác nhau

        contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 50  # Adjust this threshold as needed
        filtered_edges = np.zeros_like(mask_r)
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(filtered_edges, [contour], 0, 255, thickness=-1)
        
        img_gauss = cv2.GaussianBlur(filtered_edges, (3, 3), 0)
        fill_image = self.draw_image_with_filled_contour(img_gauss)
        edges = cv2.Canny(fill_image, 25, 200)

        kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
        merged_edges = cv2.dilate(edges, kernel, iterations=1)
        
        height = merged_edges.shape[0]
        two_thirds_height = (2 * height) // 3
        two_thirds_height = height - two_thirds_height
        img_thresholded = merged_edges.copy()
        img_thresholded[:two_thirds_height, :] = 0
        return img_thresholded
    
    def run(self):
        while not gs.exit_signal:
            if gs.rgb_frames.empty():
                time.sleep(0.1)
                continue
            image = gs.rgb_frames.get()  
            self.image = cv2.resize(image, (640, 480))
            self.mask = self.apply_canny_filter()
            # put_to_queue_no_wait_no_block(self.mask, gs.mask_img)
            gs.current_img = self.mask


            # gs.mask_img.put(self.mask)
            
            
            
            
        

    
    


    
