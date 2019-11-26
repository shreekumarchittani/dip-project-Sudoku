# Sudoku Solver using OpenCV

**Sudoku Solver solves the Sudoku puzzle from a picture taken by camera.**

It uses a collection of basic image processing techniques and K-Nearest Neighbors (KNN) algorithm for training and recognition of characters.



## Prerequisites:
  - Python 2.7  
  - OpenCV
  - Numpy


## Usage:


> git clone https://github.com/rupeshwar/Sudoku-Solver.git

> cd Sudoku-Solver

**To solve the Sudoku -**

> python solve_sudoku.py -i "path-to-input-image"

**For OCR Training -**

> python train_OCR.py -i "path-to-input-image"


## Working:
**To solve the Sudoku major steps are:**
1. Preprocessing the image
2. Finding the sudoku and do Perspective Transformation
3. Extracting the cells of the sudoku
4. Recognising the digits (OCR)
5. Solve the sudoku


#### Input Image -
<img alt="Input Sudoku" src="/Screenshots/Sudoku.jpg?raw=true" width="480px">

#### Image After Prepocessing -
<img alt="After Prepocessing" src="/Screenshots/After Prepocessing.jpg?raw=true" width="480px">

#### After Perspective Transformation -
<img alt="After Warp perspective" src="/Screenshots/After Warp perspective.jpg?raw=true" width="280px">

#### Extracting Digit - 
<img alt="Extract Digit" src="/Screenshots/Extract Digit.jpg?raw=true" width="50px">

#### Solved Sudoku -
<img alt="Solved Sudoku" src="/Screenshots/Solved Sudoku.jpg?raw=true" width="280px">
