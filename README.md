# Face Emotion Recognition

## Problem Statement 
The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms. Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge. In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention. Digital classrooms are conducted via video telephony software program (ex- Zoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to a lack of surveillance. While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms. Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

## We followed following sequence of steps to solve this problem:
* We implemented required image preprocessing techniques, image augmentation, flipping, rotating, pixel brightness transformations etc. & experimented with many predefined CNN architectures as well as a custom CNN architecture.
* Utilized transfer learning on predefined CNN, VGG, ResNet, Inception. After a lot of trial and errors we came up with some models. Highest efficiency in terms of accuracy and execution time is given by vgg16 model so we considered it as our final model.
* Our model is giving an accuracy of around 74% on train data and 67% on the test data and is robust in that it works well even with shear and tilted images even in low light. We tested the model on a local webcam and we found that it is able to detect face location and predict the right expression.
* Built an app that detects sentiment of student in online classroom from live video from a local webcam and real-time aggregated feedback to the instructor. Using this model tutors can perceive the students' emotion during online classes and react accordingly by implementing different technique of teaching.

## Model Performance Comparison
| MODEL            | TOTAL PARAMS | TRAIN ACCURACY | TEST ACCURACY |
| -----------------|--------------|----------------|---------------|
| LeNet5           | 288,387      | 60%            | 44%           |
| Alexnet          | 57,884,231   | 75%            | 59%           |
| VGG-16           | 2,430,693    | 74%            | 67%           |
| VGG-19           | 33,660,773   | 75%            | 60%           |
| Mobilenet        | 38,970,469   | 68%            | 51%           |
| Custom           | 11,272,199   | 81%            | 54%           |
| Inception        | 12,016,703   | 82%            | 47%           |
| ResNet152        | 58,650,917   | 85%            | 55%           |

## Dependencies

- Python
- Tensorflow
- Keras
- Opencv
- Streamlit

## Installation

Run project with

```Software
  Anaconda
  Command Prompt
  Jupyter notebook
  Visual Studio Code
```
    
## Run Locally

Clone the project

```bash
  git clone https://github.com/s-kp/CapstoneProject-FaceEmotionRecognition.git
```

Open Command Prompt in the project directory
```bash
  cd CapstoneProject-FaceEmotionRecognition
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start local webcam

```bash
  streamlit run app.py
```


