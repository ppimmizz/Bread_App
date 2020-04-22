from os.path import dirname
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty
import time

import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
from scipy import ndimage


class MainScreen(Screen):
    photo = StringProperty('')

    def capture(self):
        camera = self.ids['camera']
        self.photo = f"{dirname(__file__)}/IMG_{time.strftime('%Y%m%d_%H%M%S')}.png"
        camera.export_to_png(self.photo)

        # print("Captured")

        original_img = cv2.imread(self.photo)
        img = cv2.resize(original_img, (500, 500))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # normalize = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) thres_rotated =
        # cv2.adaptiveThreshold(normalize, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)

        # cv2.floodFill(thres_rotated, None, (0, 0), 255)
        # cv2.floodFill(thres_rotated, None, (0, 0), 0)
        # edges = thres_rotated
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        edges = cv2.Canny(blur, 50, 50)

        def rotated(img, edges):
            lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=80, maxLineGap=20)
            angles = []

            for x1, y1, x2, y2 in lines[0]:
                # cv2.line(img, (x1, y1), (x2, y2),0, 3)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                # print(angle)

                if angle == 0.0:
                    angles = angle
                    median_angle = np.median(angles)
                    image = ndimage.rotate(img, median_angle)
                    edges_image = edges
                    # cv2.imshow("img_NoRotated",edges_NoRotated)
                    return image, edges_image

                else:
                    angles = (0 + angle)
                    median_angle = np.median(angles)
                    image = ndimage.rotate(img, median_angle)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    normalize = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    thres_rotated = cv2.adaptiveThreshold(normalize, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY_INV,
                                                          11, 3)
                    cv2.floodFill(thres_rotated, None, (0, 0), 255)
                    cv2.floodFill(thres_rotated, None, (0, 0), 0)
                    edges_image = thres_rotated
                    # cv2.imshow("img_Rotated ",thres_rotated  )
                    return image, edges_image

        image, edges_image = rotated(img, edges)

        # คอนทัวตรงสีที่ต่างกัน #
        contours, hierarchy = cv2.findContours(edges_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # หาโลโก้ใช้คัดแยกรูปทรง #

        aplus = None
        lepan = None
        seven = None
        i = 0
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4:

                x, y, w, h = cv2.boundingRect(cnt)
                crop = image[y:y + h, x:x + w]
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                lower_orange = np.array([7, 200, 100])
                upper_orange = np.array([13, 255, 255])
                orange = cv2.inRange(hsv, lower_orange, upper_orange)
                rect = cv2.boundingRect(orange)
                x, y, w, h = rect
                logo = crop[y:y + h, x:x + w]
                if np.any(logo != 0):
                    aplus = logo
                    lepan = None
                    seven = None
                    # print (aplus)
                    # cv2.imshow('Logo_aplus', aplus)
                    # cv2.imshow('Logo_aplus{}'.format(i), aplus)
                    # i += 1



            elif 7 < len(approx) < 11:

                x, y, w, h = cv2.boundingRect(cnt)
                crop = image[y:y + h, x:x + w]
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([115, 120, 50])
                upper_blue = np.array([118, 255, 255])
                blue = cv2.inRange(hsv, lower_blue, upper_blue)
                rect = cv2.boundingRect(blue)
                x, y, w, h = rect
                logo = crop[y:y + h, x:x + w]
                if np.any(logo != 0):
                    lepan = logo
                    aplus = None
                    seven = None
                    # print ("lepan")
                    # cv2.imshow('Logo_lepan,',lepan)
                    # cv2.imshow('Logo_lepan{}'.format(i), lepan)
                    # i += 1

                else:
                    x, y, w, h = cv2.boundingRect(cnt)
                    crop = image[y:y + h, x:x + w]
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    lower_ye = np.array([18, 140, 80])
                    upper_ye = np.array([23, 200, 200])
                    ye = cv2.inRange(hsv, lower_ye, upper_ye)
                    rect = cv2.boundingRect(ye)
                    x, y, w, h = rect
                    logo = crop[y:y + h, x:x + w]
                    if np.any(logo != 0):
                        seven = logo

                        # print (aplus)
                        # cv2.imshow('Logo_seven', seven)



            elif len(approx) == 3:
                x, y, w, h = cv2.boundingRect(cnt)
                logo_triangle = image[y:y + h, x:x + w]
                # cv2.imshow("Logo_triangle",logo_triangle )

        #########################################################################################################
        # หาสี #

        color_img = original_img
        color_img = cv2.resize(color_img, (500, 500))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        reshape = color_img.reshape((color_img.shape[0] * color_img.shape[1], 3))

        def visualize_colors(cluster, centroids):
            # Get the number of different clusters, create histogram, and normalize
            labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
            (hist, _) = np.histogram(cluster.labels_, bins=labels)
            hist = hist.astype("float")
            hist /= hist.sum()

            # Create frequency rect and iterate through each cluster's color and percentage
            rect = np.zeros((50, 300, 3), dtype=np.uint8)
            colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
            start = 0
            for (percent, color) in colors:
                # print(color, "{:0.2f}%".format(percent * 100))
                end = start + (percent * 300)
                cv2.rectangle(rect, (int(start), 0), (int(end), 50), color.astype("uint8").tolist(), -1)
                start = end
            return rect

        # Find and display most dominant colors
        cluster = KMeans(n_clusters=5).fit(reshape)
        visualize = visualize_colors(cluster, cluster.cluster_centers_)
        visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
        # cv2.imshow('visualize', visualize)

        # แปลงสี BGR ไปยัง HSV
        hsv = cv2.cvtColor(visualize, cv2.COLOR_BGR2HSV)

        # หาสีตามrangeของสี

        def color_red1(visualize):
            lower_red = np.array([0, 140, 20])
            upper_red = np.array([4, 250, 255])
            crop = None
            red = cv2.inRange(hsv, lower_red, upper_red)
            if cv2.countNonZero(red) == 0:
                return 0
            else:
                rect = cv2.boundingRect(red)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('red', crop)
                return 1

        def color_red2(visualize):
            lower_red = np.array([176, 180, 20])
            upper_red = np.array([180, 255, 255])
            crop = None
            red = cv2.inRange(hsv, lower_red, upper_red)
            if cv2.countNonZero(red) == 0:
                return 0
            else:
                rect = cv2.boundingRect(red)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('red', crop)
                return 1

        def color_purple(visualize):
            lower_purple = np.array([150, 50, 20])
            upper_purple = np.array([200, 255, 255])
            purple = cv2.inRange(hsv, lower_purple, upper_purple)
            crop = None
            if cv2.countNonZero(purple) == 0:
                return 0
            else:
                rect = cv2.boundingRect(purple)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('purple', crop)
                return 1

        def color_green(visualize):
            lower_green = np.array([26, 90, 20])
            upper_green = np.array([69, 255, 255])
            green = cv2.inRange(hsv, lower_green, upper_green)
            # mask = cv2.bitwise_and(visualize,visualize, mask= green)
            crop = None
            if cv2.countNonZero(green) == 0:
                return 0
            else:
                rect = cv2.boundingRect(green)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('green', crop)
                return 1

        def color_orange(visualize):
            lower_orange = np.array([4, 150, 100])
            upper_orange = np.array([17, 255, 255])
            orange = cv2.inRange(hsv, lower_orange, upper_orange)
            crop = None
            if cv2.countNonZero(orange) == 0:
                return 0
            else:
                rect = cv2.boundingRect(orange)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('orange', crop)
                return 1

        def color_blue(visualize):
            lower_blue = np.array([90, 50, 20])
            upper_blue = np.array([124, 255, 255])
            blue = cv2.inRange(hsv, lower_blue, upper_blue)
            crop = None
            if cv2.countNonZero(blue) == 0:
                return 0
            else:
                rect = cv2.boundingRect(blue)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('blue', crop)
                return 1

        def color_yellow(visualize):
            lower_yellow = np.array([18, 60, 50])
            upper_yellow = np.array([25, 255, 255])
            yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            crop = None
            if cv2.countNonZero(yellow) == 0:
                return 0
            else:
                rect = cv2.boundingRect(yellow)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('yellow', crop)
                return 1

        def color_violet(visualize):
            lower_violet = np.array([125, 100, 20])
            upper_violet = np.array([140, 255, 255])
            violet = cv2.inRange(hsv, lower_violet, upper_violet)
            crop = None
            if cv2.countNonZero(violet) == 0:
                return 0
            else:
                rect = cv2.boundingRect(violet)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('violet', crop)
                return 1

        def color_brown(visualize):
            lower_brown = np.array([5, 75, 0])
            upper_brown = np.array([10, 255, 100])
            brown = cv2.inRange(hsv, lower_brown, upper_brown)
            crop = None
            if cv2.countNonZero(brown) == 0:
                return 0
            else:
                rect = cv2.boundingRect(brown)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('brown', crop)
                return 1

        def color_beige(visualize):
            lower_beige = np.array([10, 50, 0])
            upper_beige = np.array([17, 120, 255])
            beige = cv2.inRange(hsv, lower_beige, upper_beige)
            crop = None
            if cv2.countNonZero(beige) == 0:
                return 0
            else:
                rect = cv2.boundingRect(beige)
                x, y, w, h = rect
                crop = visualize[y:y + h, x:x + w]
                # cv2.imshow('beige', crop)
                return 1

        red1 = color_red1(visualize)
        red2 = color_red2(visualize)
        green = color_green(visualize)
        orange = color_orange(visualize)
        blue = color_blue(visualize)
        yellow = color_yellow(visualize)
        violet = color_violet(visualize)
        brown = color_brown(visualize)
        purple = color_purple(visualize)
        beige = color_beige(visualize)

        cv2.waitKey(0)

        def result():

            if aplus is None and lepan is None and seven is None:
                result_text = " ไม่พบข้อมูลในฐานข้อมูล กรุณาถ่ายใหม่อีกครั้ง "
                print(result_text)
                return result_text

            elif aplus is not None:

                if red1 == 1 or red2 == 1:
                    result_text = "ชื่อยี่ห้อ : Aplus \n รสชาติ : ขนมปังไส้ถั่วแดง \n ข้อมูลสำหรับผู้แพ้อาหาร : " \
                                  "มีแป้งสาลี ผลิตภัณฑ์จากนมและถั่วเหลือง \n "
                    print(result_text)
                    return result_text

                elif green == 1:
                    result_text = "ชื่อยี่ห้อ : Aplus \n รสชาติ : ขนมปังไส้สังขยา \n ข้อมูลสำหรับผู้แพ้อาหาร : " \
                                  "มีแป้งสาลี ไข่ ผลิตภัณฑ์จากนมและถั่วเหลือง \n "
                    print(result_text)
                    return result_text

                elif violet == 1:
                    result_text = "ชื่อยี่ห้อ : Aplus \n รสชาติ : ขนมปังไส้เผือก \n ข้อมูลสำหรับผู้แพ้อาหาร : " \
                                  "มีแป้งสาลี ผลิตภัณฑ์จากนมและถั่วเหลือง \n"
                    print(result_text)
                    return result_text

                elif brown == 1:
                    result_text = "ชื่อยี่ห้อ : Aplus \n รสชาติ : ขนมปังเพรสช็อกโกแลต \n ข้อมูลสำหรับผู้แพ้อาหาร : " \
                                  "มีแป้งสาลี ไข่และนม \n"
                    print(result_text)
                    return result_text

                elif orange == 1:
                    result_text = "ชื่อยี่ห้อ : Aplus \n รสชาติ : ขนมปังกรอบหน้าเนย \n ข้อมูลสำหรับผู้แพ้อาหาร : " \
                                  "มีแป้งสาลี ผลิตภัณฑ์จากนมและถั่วเหลือง \n"
                    print(result_text)
                    return result_text

                else:
                    result_text = " ไม่พบข้อมูลในฐานข้อมูล กรุณาถ่ายใหม่อีกครั้ง "
                    print(result_text)
                    return result_text

            elif lepan is not None:

                if yellow == 1 and (brown == 1 or purple == 1):
                    result_text = "ชื่อยี่ห้อ : Lepan \n รสชาติ : ดับเบิ้ลแซนด์วิชกระเป๋า ไส้หมูหยองมายองเนส  และ " \
                                  "ไส้ปูอัดมายองเนส \n ข้อมูลสำหรับผู้แพ้อาหาร : มีแป้งสาลี ไข่ นม ปลา \n" \
                                  " ผลิตภัณฑ์จากถั่วเหลือง ผลิตภัณฑ์จากสัตว์น้ำที่มีเปลือกแข็ง และ อาจมีอัลมอนด์ "
                    print(result_text)
                    return result_text

                elif purple == 1:
                    result_text = " ชื่อยี่ห้อ : Lepan \n รสชาติ : ดับเบิ้ลแซนด์วิชกระเป๋า ไส้ปูอัดมายองเนส และ " \
                                  "ไส้ทูน่ามายองเนส    " \
                                  "\n ข้อมูลสำหรับผู้แพ้อาหาร :  มีแป้งสาลี ไข่ นม ปลา \n" \
                                  "ผลิตภัณฑ์จากถั่วเหลือง ผลิตภัณฑ์จากสัตว์น้ำที่มีเปลือกแข็ง และ อาจมีอัลมอนด์ \n"
                    print(result_text)
                    return result_text

                elif orange == 1 or beige == 1:
                    result_text = " ชื่อยี่ห้อ : Lepan \n รสชาติ : แซนด์วิชกระเป๋า ไส้ปูอัดมายองเนส      \n " \
                                  "ข้อมูลสำหรับผู้แพ้อาหาร : มีแป้งสาลี ไข่ นม ปลา ผลิตภัณฑ์จากถั่วเหลือง   \n" \
                                  " ผลิตภัณฑ์จากสัตว์น้ำที่มีเปลือกแข็ง และ อาจมีอัลมอนด์  \n"
                    print(result_text)
                    return result_text

                elif yellow == 1:
                    result_text = " ชื่อยี่ห้อ : Lepan \n รสชาติ : แซนด์วิชกระเป๋า ไส้หมูหยองมายองเนส     \n " \
                                  "ข้อมูลสำหรับผู้แพ้อาหาร : มีแป้งสาลี ไข่ นม \n" \
                                  "ผลิตภัณฑ์จากถั่วเหลือง และอาจมี ปลา อัลมอนด์ "

                    print(result_text)
                    return result_text

                elif blue == 1:
                    result_text = " ชื่อยี่ห้อ : Lepan \n รสชาติ : แซนด์วิชกระเป๋า ไส้ทูน่า  \n " \
                                  "ข้อมูลสำหรับผู้แพ้อาหาร : มีแป้งสาลี ปลา นม ไข่ \n ผลิตภัณฑ์จากถั่วเหลือง และ " \
                                  "อาจมีอัลมอนด์ "
                    print(result_text)
                    return result_text

                else:
                    result_text = " ไม่พบข้อมูลในฐานข้อมูล กรุณาถ่ายใหม่อีกครั้ง "
                    print(result_text)
                    return result_text

            elif seven is not None:

                if blue == 1:
                    result_text = "ชื่อยี่ห้อ : EZY BAKE \n รสชาติ : พายไส้มะพร้าว  \n ข้อมูลสำหรับผู้แพ้อาหาร : " \
                                  "มีแป้งสาลี ไข่ เลซิตินจากถั่วเหลือง \n และอาจมี นม ปลา อัลมอนด์  \n "
                    print(result_text)
                    return result_text

                elif purple == 1:
                    result_text = "ชื่อยี่ห้อ : EZY BAKE \n รสชาติ : พายไส้เผือกมะพร้าวอ่อน  \n " \
                                  "ข้อมูลสำหรับผู้แพ้อาหาร : มีแป้งสาลี ไข่ ผลิตภัณฑ์จากนม \n เลซิตินจากถั่วเหลือง " \
                                  "และอาจมี ปลา อัลมอนด์ \n"
                    print(result_text)
                    return result_text


                elif yellow == 1:
                    result_text = " ชื่อยี่ห้อ : EZY BAKE \n รสชาติ : พายไส้ข้าวโพด  " \
                                  "\n ข้อมูลสำหรับผู้แพ้อาหาร :  มีแป้งสาลี ไข่ นม ผลิตภัณฑ์จากนม \n " \
                                  "เลซิตินจากถั่วเหลือง และ อาจมี ปลา อัลมอนด์  \n"
                    print(result_text)
                    return result_text

                elif brown == 1:
                    result_text = "ชื่อยี่ห้อ : EZY BAKE \n รสชาติ : เดนิช ไส้ช็อกโกแลต    " \
                                  "\n ข้อมูลสำหรับผู้แพ้อาหาร : มีแป้งสาลี ไข่ \n" \
                                  "นมและผลิตภัณฑ์จากนมถั่วเหลืองและเลซิตินจากถั่วเหลือง\n" \
                                  "  อัลมอนด์ และ อาจมี ปลา ซัลไฟต์ \n"
                    print(result_text)
                    return result_text

                elif beige == 1:
                    result_text = " ชื่อยี่ห้อ : EZY BAKE \n รสชาติ : พายไส้แฮมเห็ดไวท์ซอส        \n " \
                                  "ข้อมูลสำหรับผู้แพ้อาหาร : มีแป้งสาลี ไข่ นมและผลิตภัณฑ์จากนม " \
                                  "\n ผลิตภัณฑ์จากถั่วเหลือง เลซิตินจากถั่วเหลือง \n และ อาจมี ปลา อัลมอนด์    \n"
                    print(result_text)
                    return result_text

                else:
                    result_text = " ไม่พบข้อมูลในฐานข้อมูล กรุณาถ่ายใหม่อีกครั้ง "
                    print(result_text)
                    return result_text

        self.manager.get_screen('show_result').Result_Breads = result()


class ShowResult(Screen):
    Result_Breads = StringProperty()


    def on_pre_enter(self, *args):
        self.ids.img.source = self.manager.ids.main_screen.photo


class ScreenManagement(ScreenManager):
    pass


Builder.load_file('GUI.kv')


class Interface(App):

    def build(self):
        return ScreenManagement()


Project_app = Interface()
Project_app.run()
