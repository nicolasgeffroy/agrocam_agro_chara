### 1️⃣ Segmentation

- **Input:** RGB vineyard images (1920×1080, JPEG).  
- **Model:** MobileNetV3 (PyTorch) trained with **Focal Loss**.  
- **Classes:** Leaf, Trunk, Inter-row, Irrigation Sheath.  
- **Metrics:** IoU, Sensitivity, Specificity.  
- **Performance:**  
  - IoU: *Leaves = 0.87*, *Inter-row = 0.92*, *Trunk = 0.58*, *Sheath = 0.42*.