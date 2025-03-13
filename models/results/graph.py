import matplotlib.pyplot as plt
import numpy as np

# Model names and corresponding accuracies
models = ["ResNET (None)", "ResNET (CLAHE)", "ResNET (Sharpness)", "YOLO (Ultralytics Aug)", 
          "ResNET (Gaussian Blur)", "ResNET (Reinhard)", "ViT (Did not train)"]

accuracies = [0.916, 0.910, 0.900, 0.891, 0.890, 0.857, 0]  # ViT didn't train, so accuracy is 0

# Set up figure and axis
plt.figure(figsize=(10, 5))
bars = plt.bar(models, accuracies, color=['wheat', 'gainsboro', 'powderblue', 'pink', 'mediumspringgreen', 'gray'])

# Add text labels above bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.3f}", ha='center', va='bottom')

# Labels and title
plt.xlabel("Model & Preprocessing")
plt.ylabel("Test Accuracy")
plt.title("Model Accuracy on Test Set with Different Pre-Processing Techniques")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1)  # Accuracy range from 0 to 1
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show plot
plt.show()
