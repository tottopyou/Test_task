# CV Engineer – Test Assignment

This project contains solutions to two Computer Vision tasks as outlined in the assignment:

1. **Image Matching – Position Detection**
2. **Road Detection in Images**

---

## Streamlit Visualization

For visualization in this project, we use **Streamlit**, which allows for easy testing and interaction with the implemented tasks. All results, including intermediate outputs, are visualized directly in the Streamlit interface, making it straightforward to understand and experiment with the solutions.

---

## Task 1: Image Matching – Position Detection

### Objective
Detect the position of a smaller image within a larger image using Template Matching.

### Approach

1. **Streamlit Interface:**
   - Users upload the smaller (template) image and the larger image (search area).
   - Users can choose either a default scale array or enter custom scales for better results.
2. **Template Matching:**
   - Multiple scales are tested to account for size differences.
   - Normalized correlation is used to measure similarity.
3. **Output:**
   - Coordinates of the top-left corner of the smaller image in the larger image.
   - A visualization of the matched region is displayed in the Streamlit app.

### Key Features

- **Scale Selection:** Users can choose between default scales or input custom scales for better matching accuracy.

### Results

For given test images:
| small image | large image | result |
| ------- | ------- | -------- |
| ![image](https://github.com/user-attachments/assets/2c656db4-5963-4d3d-ba78-401097eb6429)| ![image](https://github.com/user-attachments/assets/0f9eaa24-a11a-498c-b699-0a709c10ffbf)| ![image](https://github.com/user-attachments/assets/2936e0af-934f-42b7-9820-51b28aaa3603)|

#### Top-left corner coordinates of the smaller image in the larger image: (103, 197)
---

## Task 2: Road Detection in Image

### Objective
Detect and visualize roads in an image using image segmentation techniques.

### Approach

1. **Streamlit Interface:**
   - Users upload an image and choose scaling options (original size, 2x, 4x, or 8x).
   - Larger scaling factors generally yield better results.
   - Users can select either **CPU** or **GPU** for processing.
   - A button is provided to test the availability of a GPU for accelerated computation.
2. **Model Integration:** Utilized the [DeepSegmentor](https://github.com/yhlleo/DeepSegmentor) model for road detection.
3. **Segmentation:**
   - The model processes the uploaded image and generates a segmented output highlighting roads.
4. **Output:**
   - Results are visualized in the Streamlit interface, including the scaled input and the segmented output.

### Key Features

- **Image Scaling:** Provides options to scale the uploaded image for better segmentation results.
- **Device Selection:** Allows users to choose between CPU or GPU for processing, with GPU availability testing.
- **Streamlit Visualization:** Displays the uploaded image, scaling options, and segmented results interactively.

### Results

For given test images:
| input image | result |
| ------- | ------- |
|  ![large_image_8xScope](https://github.com/user-attachments/assets/1561214d-a40f-48a8-a2ef-50f0dd7b07fe)| ![large_image_label_pred](https://github.com/user-attachments/assets/bfa089eb-dcc7-4f99-bbfc-1d8f6381416a)| 

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tottopyou/Test_task.git
   cd Test_task
   ```

2. Build and start the project using Docker Compose:
   ```bash
   docker-compose build
   docker-compose up
   ```

3. Open the Streamlit page:
   ![image](https://github.com/user-attachments/assets/663c939f-9f7f-442f-b380-56cf884fcf37)

The application will provide a unified interface for both tasks:

- **Task 1:** Upload the smaller and larger images, choose the scale mode, and view the detected region.
- **Task 2:** Upload an image, select the scaling factor, and view the segmented roads in the output.

### Warning

> **⚠️ Warning:** If you choose an 8x scale, it may take a significant amount of time to generate results, even when using a GPU.

---

## Challenges and Solutions

### Task 1

- **Challenge:** The primary issue was managing the varying quality and size of the input images. The smaller image was not proportional to the larger image, making direct template matching challenging.
- **Solution:** After experimenting with various OpenCV methods, such as `Difference-of-Gaussians`, and `Scale-Invariant Feature Transforms`, it became clear that scaling the smaller image to match the larger image’s proportions was essential. The implemented solution involved dynamically resizing the smaller image across multiple scales and using `cv2.TM_CCOEFF_NORMED` to determine the best match. This approach, combined with customizable scaling options, provided precise and reliable results. Allowing users to input custom scales further enhanced the accuracy and flexibility of the solution, yielding consistently better results with appropriate adjustments to the smaller image's size.

### Task 2

- **Challenge:** The complexity of detecting roads in images stemmed from the quality and scale of the objects in the photos. Traditional lightweight methods, such as edge detection and thresholding, were insufficient due to the intricate nature of the task.
- **Solution:** Recognizing the limitations of simpler approaches, a pre-trained model was necessary. After considerable research, DeepSegmentor ([https://github.com/yhlleo/DeepSegmentor](https://github.com/yhlleo/DeepSegmentor)) was selected for its robust road detection capabilities, including centerline detection. However, challenges remained. The model did not natively support JPG images, requiring conversion to PNG format. Additionally, scaling the input images to larger sizes significantly improved detection accuracy. GPU configuration was also addressed to ensure optimal performance during processing. By implementing these adjustments, the solution effectively utilized the DeepSegmentor model to achieve high-quality road detection results with scalable input images.

## References to authors of DeepSegmentor model:
```
@article{liu2019deepcrack,
  title={DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
  journal={Neurocomputing},
  volume={338},
  pages={139--153},
  year={2019},
  doi={10.1016/j.neucom.2019.01.036}
}

@article{liu2019roadnet,
  title={RoadNet: Learning to Comprehensively Analyze Road Networks in Complex Urban Scenes from High-Resolution Remotely Sensed Images},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xia, Menghan and Wang, Xingbo and Liu, Yuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={57},
  number={4},
  pages={2043--2056},
  year={2019},
  doi={10.1109/TGRS.2018.2870871}
}
```

