<h1>Smart Crop Monitoring System</h1>

This work presents an integrated deep learning system for precision agriculture, specifically targeting mango orchards. The system leverages aerial imagery captured by drones and processes it through a series of specialized models to perform comprehensive diagnostics. It combines object detection, image classification, and vegetation index analysis to identify mango trees, assess stress levels through NDVI heatmaps, detect leaf diseases, classify pest infestations, and identify weed growth. The detection pipeline uses YOLOv8 for tree and weed detection, a custom ResNet-based model for NDVI stress classification, and EfficientNet and MobileNetV2 for disease and pest classification, respectively. The models are integrated into a unified Flask web interface that allows image upload, automated processing, and real-time result display. This tool aims to empower farmers, researchers, and agronomists by providing actionable insights to improve crop management and yield outcomes.


models folder: [https://drive.google.com/drive/folders/1P-GwHORBywvRprBi8a7EF3cnIqZpOThk?usp=sharing]
