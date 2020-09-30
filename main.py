import cv2
import numpy as np

img = cv2.imread("Resources/pina.jpg", cv2.IMREAD_UNCHANGED)
img2 = cv2.imread("Resources/blueMe.png", cv2.IMREAD_UNCHANGED)
imgGray = cv2.imread("Resources/pina.jpg", 0)

def blueScreen():
    image_copy = np.copy(img2)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    # color threshold
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([120, 80, 255])

    # creating the mask
    mask = cv2.inRange(image_copy, lower_blue, upper_blue)

    # Masked image
    masked_image = np.copy(image_copy)
    masked_image[mask != 0] = [0, 0, 0]
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

    # Creating the background
    backgroundImg = cv2.imread("Resources/pina.jpg")
    # backgroundImg = cv2.cvtColor(backgroundImg, cv2.COLOR_BGR2RGB)

    crop_back = backgroundImg[0:480, 0:852]
    crop_back[mask == 0] = [0, 0, 0]

    # Final image
    final_image = crop_back + masked_image

    cv2.imshow("BlueScreen", final_image)
# Got from: https://github.com/tejakummarikuntla/blue-screen-effect-OpenCV/blob/master/blue-screen-effect.ipynb

# Photo effects

def bright():
    print("Enter a intensity factor: ")
    intensity = int(input())
    output = np.zeros(img.shape,np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                output[y,x,c] = np.clip(1.5*img[y,x,c] + intensity, 0, 255)
    return output

# Got from: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

def contrast():
    print("Enter a contrast factor: [1.0 - 3.0]")
    intensity = float(input())
    output = np.zeros(img.shape, np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                output[y, x, c] = np.clip(intensity * img[y, x, c], 0, 255)
    return output

# Got from: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

def gammaCorrection():
    gamma = 0.5
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(img, lookUpTable)
    ## [changing-contrast-brightness-gamma-correction]

    img_gamma_corrected = cv2.hconcat([img, res])
    # cv2.imshow("Gamma correction", res)
    return res

# Got from: https://github.com/opencv/opencv/blob/3.4/samples/python/tutorial_code/imgProc/changing_contrast_brightness_image/changing_contrast_brightness_image.py

def solarization():
    solarization_const = 2 * np.pi / 255

    look_up_table = np.ones((256, 1), dtype='uint8') * 0

    for i in range(256):
        look_up_table[i][0] = np.abs(np.sin(i * solarization_const)) * 100

    img_sola = cv2.LUT(imgGray, look_up_table)

    # cv2.imshow("Solarization", img_sola)
    return img_sola

# Got from: https://github.com/umentu/opencv

def histogramEqualization():
    # Create the equalize histogram from the grayscale image
    img_eqh = cv2.equalizeHist(imgGray)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(imgGray)

    # cv2.imshow("Original Gray", imgGray)
    # cv2.imshow("Histogram equalization", img_eqh)
    # cv2.imshow("Contrast Limited Adaptive Histogram Equalization", cl1)
    return img_eqh

# Got from: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html

def resizeAnimation():
    (h, w, d) = img.shape

    for x in range(0, 5):
        scaleFactor = 0.25
        for i in range(0, 5):
            r = int(scaleFactor * w)  # Resize measure
            dim = (r, int(h * (r / w)))

            #Here comes the magic
            if x == 0:
                # Bright es un filtro muy pesasdo
                # resized = cv2.resize(bright(), dim)

                resized = cv2.resize(solarization(), dim)
                cv2.imshow("Aspect Ratio Resize with solarization", resized)
            elif x == 1:
                # contrast es casi lo mismo, y es un filtro muy pesado
                # resized = cv2.resize(contrast(), dim)

                resized = cv2.resize(gammaCorrection(), dim)
                cv2.imshow("Aspect Ratio Resize with gamma", resized)
            elif x == 2:
                resized = cv2.resize(histogramEqualization(), dim)
                cv2.imshow("Aspect Ratio Resize with histogram equalization", resized)
            cv2.waitKey(1000)
            scaleFactor = scaleFactor + 0.25

# imgB = bright()
# cv2.imshow('brightImage.jpg', imgB)
cv2.imshow("Image  original", img)
# blueScreen()
# imgC = contrast()
# cv2.imshow("Contrasted image", imgC)
# gammaCorrection()
# solarization()
# histogramEqualization()
resizeAnimation()

cv2.waitKey(0)