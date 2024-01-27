import cv2
import numpy as np
import global_storage as gs

class ObjectFinder:
    def __init__(self):
      self.minsize = 15

    def filter_signs_by_color(self, image):
        """Lọc các đối tượng màu đỏ và màu xanh dương - Có thể là biển báo.
            Ảnh đầu vào là ảnh màu BGR
        """
        # Chuyển ảnh sang hệ màu HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Lọc màu đỏ cho stop và biển báo cấm
        # simulator
        # lower1, upper1 = np.array([77, 70, 20]), np.array([141, 255, 136]) 
        # lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
        
        # Test trong truong
        lower1, upper1 = np.array([27, 118, 40]), np.array([179, 255, 255])
        
        # real world
        # lower1, upper1 = np.array([0, 100, 110]), np.array([179, 255, 255])
        
        # Test trong truong
        # lower2, upper2 = np.array([10, 67, 0]), np.array([179, 255, 158])
        
        
        lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
        mask_1 = cv2.inRange(image, lower1, upper1) # dải màu đỏ thứ nhất
        mask_2 = cv2.inRange(image, lower2, upper2) # dải màu đỏ thứ hai
        mask_r = cv2.bitwise_or(mask_1, mask_2) # kết hợp 2 kết quả từ 2 dải màu khác nhau

        # Lọc màu xanh cho biển báo điều hướng
        lower3, upper3 = np.array([85, 50, 200]), np.array([135, 250, 250])
        lower4, upper4 = np.array([96, 69, 65]), np.array([110, 245, 133]) #biển xanh dương nhỏ 
        mask_3 = cv2.inRange(image, lower3,upper3)
        mask_4 = cv2.inRange(image, lower4,upper4)
        mask_b = cv2.bitwise_or(mask_3, mask_4)
        # mask_b = cv2.inRange(image, lower3,upper3)

        # Kết hợp các kết quả
        mask_final  = cv2.bitwise_or(mask_r,mask_b)
        if gs.show_hsvmask:
            cv2.imshow("Traffic signs", mask_final)
            cv2.waitKey(1)
        return mask_final


    def get_boxes_from_mask(self, img):
        """Tìm kiếm hộp bao biển báo
        """
        mask = self.filter_signs_by_color(img)
        bboxes = []

        nccomps = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        numLabels, labels, stats, centroids = nccomps
        # print("Okeoeoek: ", numLabels)
        im_height, im_width = mask.shape[:2]
        for i in range(numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # Lọc các vật quá nhỏ, có thể là nhiễu
            if w < self.minsize or h < self.minsize:
                continue
            # Lọc các vật quá lớn
            if w > 0.8 * im_width or h > 0.8 * im_height:
                continue
            # Loại bỏ các vật có tỷ lệ dài / rộng quá khác biệt
            if w / h > 2.0 or h / w > 2.0:
                continue
            bboxes.append([x, y, w, h])
        return bboxes