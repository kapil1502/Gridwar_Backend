import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from inference_sdk import InferenceHTTPClient

# def enhance_contrast(image):
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl,a,b))
#     final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#     return final

# def sharpen_image(image):
#     kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#     sharpened = cv2.filter2D(image, -1, kernel)
#     return sharpened

# def remove_background(image):
#     mask = np.zeros(image.shape[:2], np.uint8)
#     bgdModel = np.zeros((1,65), np.float64)
#     fgdModel = np.zeros((1,65), np.float64)
#     rect = (50,50,450,290)
#     cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
#     mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
#     image = image*mask2[:,:,np.newaxis]
#     return image

# def correct_perspective(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
#     if lines is not None:
#         for rho, theta in lines[0]:
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a*rho
#             y0 = b*rho
#             x1 = int(x0 + 1000*(-b))
#             y1 = int(y0 + 1000*(a))
#             x2 = int(x0 - 1000*(-b))
#             y2 = int(y0 - 1000*(a))
#             cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 2)
#     return image

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise removal
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    # Dilation and erosion
    #kernel = np.ones((3,3), np.uint8)
    #dilated = cv2.dilate(denoised, kernel, iterations=1)
    #eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Edge enhancement
    #edges = cv2.Canny(eroded, 100, 200)
    #enhanced = cv2.addWeighted(eroded, 0.8, edges, 0.2, 0)
    
    return denoised

def assess_freshness(img_path):
    # Read the image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image for faster processing
    resized = cv2.resize(image, (300, 300))

    # Flatten the image
    pixels = resized.reshape(-1, 3)

    # Use K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pixels)

    # Get the dominant colors
    colors = kmeans.cluster_centers_

    # Calculate color vibrancy (using standard deviation of colors)
    color_vibrancy = np.std(colors)

    # Calculate texture (using edge detection)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    plt.imshow(edges)
    plt.show()
    texture_score = np.sum(edges) / (300 * 300)  # Normalize by image size
    
    #Assigning Freshness_score based on color vibrancy and texture
    freshness_score = color_vibrancy - 2*texture_score

    print(freshness_score)

    # Now using freshness_score to give final verdict
    if freshness_score > 70 and freshness_score < 100:
        return "Fresh"
    elif freshness_score > 45 and freshness_score < 70:
        return "Moderately Fresh"
    else:
        return "Not Fresh"

# def analyze_apple_freshness(image_path):
#     # Read the image
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Resize image for faster processing
#     resized = cv2.resize(image, (300, 300))
    
#     # Flatten the image
#     pixels = resized.reshape(-1, 3)

#     # Use K-means clustering to find dominant colors
#     kmeans = KMeans(n_clusters=5)
#     kmeans.fit(pixels)

#     # Get the dominant colors
#     colors = kmeans.cluster_centers_

#     # Calculate the variation in red channel
#     red_values = colors[:, 0]  # Red channel values
#     color_variation = np.std(red_values)

#     # Define thresholds for freshness
#     low_variation_threshold = 20
#     high_variation_threshold = 50

#     # Assess freshness based on color variation
#     if color_variation < low_variation_threshold:
#         return "Rotten", color_variation
#     elif color_variation > high_variation_threshold:
#         return "Moderately Fresh", color_variation
#     else:
#         return "Fresh", color_variation
    
# image_path = r"D:\Gridwar\apple-fresh.jpg"
# print(assess_freshness(image_path=image_path))

# # # Read image
# # image_path = r"D:\Gridwar\oatsTestImage.jpg"
# # img = cv2.imread(image_path)

# # img2 = preprocess_image(img)

# # plt.figure(figsize=(12, 12))
# # plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
# # plt.axis('off')
# # plt.title("Detected Text")
# # plt.show()

img_path = r'D:\Gridwar\fr-egg.jpg'
print(assess_freshness(img_path))

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title("Detected Text")
# plt.show()