# Handwriting Digit Calculator

An AI-powered calculator that recognizes handwritten digits and mathematical operators to compute results in real-time. It uses contour detection and a Convolutional Neural Network (CNN) model for predictions.

---

## Features

* Recognizes digits `0-9` and operators `+`, `-`, `×`, `/`
* Supports decimal point `.` and parentheses `( )`
* Supports exponentiation `**` and floor division `//`
* Real-time predictions with an interactive GUI
* Visualizes contours and predictions for clarity

---

## Technology Stack

* **Python**
* **TensorFlow (Keras)** – for CNN model creation and prediction
* **CustomTkinter** – for GUI
* **OpenCV** – for image processing and contour detection
* **Pillow (PIL)** – to handle canvas input
* **NumPy & Pandas** – for data handling


## Installation

1. Clone the repository:

```bash
git clone https://github.com/srush17/Handwriting-Digit-Calculator.git
```

2. Navigate to the project folder:

```bash
cd Handwriting-Digit-Calculator
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Usage

* Run the Jupyter Notebook:

```bash
jupyter notebook MAIN.ipynb
```

* Or run the Python script:

```bash
python MAIN.py
```

* Draw digits and operators on the canvas. The calculator will detect and evaluate the expression automatically.

---

## Data Sources

* **MNIST Dataset**: [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
* **Symbols Dataset**: [IrFanchahyadi GitHub](https://github.com/irfanchahyadi/Handwriting-Calculator/blob/master/src/dataset/data.pickle)

---

## License

This project is open-source and free to use.

---

## Author

**Srushti Bankar**

* Email: [srushtibankar17@gmail.com]
* LinkedIn: [https://www.linkedin.com/in/srushti-bankar-09a8a2224]
* Instagram: [https://www.instagram.com/srush_17_]
* GitHub: [https://github.com/srush17]
