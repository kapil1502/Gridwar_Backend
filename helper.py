import cv2
import typing_extensions as typing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise removal
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    # Dilation and erosion
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Edge enhancement
    edges = cv2.Canny(eroded, 100, 200)
    enhanced = cv2.addWeighted(eroded, 0.8, edges, 0.2, 0)
    
    return enhanced

def correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 2)
    return image

def assess_freshness(img):
    # Read the image
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    texture_score = np.sum(edges) / (300 * 300)  # Normalize by image size
    
    #Assigning Freshness_score based on color vibrancy and texture
    freshness_score = color_vibrancy - 2*texture_score

    level = None

    # Now using freshness_score to give final verdict
    if freshness_score > 70 and freshness_score < 100:
        level="Fresh"
    elif freshness_score > 45 and freshness_score < 70:
        level="Moderately Fresh"
    else:
        level="Not Fresh"
    
    return {
        'freshness_score': int(freshness_score),
        'level': level
    }

# Defining a new Class for enforcing the structured format of reponse from LLM side for Product information extracted from the image
class ProductDescription(typing.TypedDict):
    brand_name: str
    product_name: str
    manufacturing_date: str
    expiry_date: str
    weight: str
    price: str
    description: str
    Ingredients: list[str]