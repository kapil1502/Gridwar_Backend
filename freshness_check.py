import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ProduceFreshnessAnalyzer:
    def __init__(self):
        self.produce_rules = {
            'apple': {'color_range': ([0, 100, 100], [10, 255, 255]), 'shape': 'round', 'texture_threshold': 0.1},
            'banana': {'color_range': ([20, 100, 100], [30, 255, 255]), 'shape': 'curved', 'texture_threshold': 0.05},
            'tomato': {'color_range': ([0, 100, 100], [10, 255, 255]), 'shape': 'round', 'texture_threshold': 0.08},
            'lettuce': {'color_range': ([35, 100, 100], [85, 255, 255]), 'shape': 'leafy', 'texture_threshold': 0.15},
            # Add more produce types here
        }

    def analyze_freshness(self, image_path, produce_type):
        if produce_type not in self.produce_rules:
            raise ValueError(f"No rules defined for {produce_type}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (300, 300))
        
        color_score = self._analyze_color(resized, produce_type)
        texture_score = self._analyze_texture(resized, produce_type)
        shape_score = self._analyze_shape(resized, produce_type)
        
        # Combine scores (you can adjust weights as needed)
        freshness_score = (color_score * 0.5 + texture_score * 0.3 + shape_score * 0.2) * 100
        
        if freshness_score > 70:
            return "Fresh", freshness_score
        elif freshness_score > 50:
            return "Moderately Fresh", freshness_score
        else:
            return "Not Fresh", freshness_score

    def _analyze_color(self, image, produce_type):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_color, upper_color = self.produce_rules[produce_type]['color_range']
        mask = cv2.inRange(hsv_image, np.array(lower_color), np.array(upper_color))
        color_percentage = (np.sum(mask) / 255) / (300 * 300)
        return color_percentage

    def _analyze_texture(self, image, produce_type):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        texture_score = np.sum(edges) / (300 * 300)
        threshold = self.produce_rules[produce_type]['texture_threshold']
        return 1 - min(texture_score / threshold, 1)  # Inverse score, less texture is fresher

    def _analyze_shape(self, image, produce_type):
        # This is a simplified shape analysis and can be expanded
        expected_shape = self.produce_rules[produce_type]['shape']
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if expected_shape == 'round':
            circularity = 4 * np.pi * cv2.contourArea(contours[0]) / (cv2.arcLength(contours[0], True) ** 2)
            return min(circularity, 1)  # 1 is a perfect circle
        elif expected_shape == 'curved':
            # Simplified banana shape analysis
            rect = cv2.minAreaRect(contours[0])
            aspect_ratio = max(rect[1]) / min(rect[1])
            return min((aspect_ratio - 1) / 2, 1)  # Adjust as needed
        elif expected_shape == 'leafy':
            # Simplified leafy shape analysis
            area = cv2.contourArea(contours[0])
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            return solidity  # Lower solidity indicates more leafy structure
        else:
            return 1  # Default score if shape is not specifically analyzed

    def visualize_analysis(self, image_path, produce_type):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (300, 300))
        
        plt.figure(figsize=(12, 8))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(resized)
        plt.title("Original Image")
        plt.axis('off')
        
        # Color mask
        hsv_image = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        lower_color, upper_color = self.produce_rules[produce_type]['color_range']
        mask = cv2.inRange(hsv_image, np.array(lower_color), np.array(upper_color))
        plt.subplot(2, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Color Mask")
        plt.axis('off')
        
        # Edge detection
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        plt.subplot(2, 2, 3)
        plt.imshow(edges, cmap='gray')
        plt.title("Edge Detection")
        plt.axis('off')
        
        # Shape analysis
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        shape_image = resized.copy()
        cv2.drawContours(shape_image, contours, -1, (0, 255, 0), 2)
        plt.subplot(2, 2, 4)
        plt.imshow(shape_image)
        plt.title("Shape Analysis")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
analyzer = ProduceFreshnessAnalyzer()
image_path = r"D:\Gridwar\rotten-apple.jpg"
produce_type = "apple"  # Change this to the type of produce you're analyzing

freshness, score = analyzer.analyze_freshness(image_path, produce_type)
print(f"The {produce_type} appears to be: {freshness}")
print(f"Freshness score: {score:.2f}")

analyzer.visualize_analysis(image_path, produce_type)