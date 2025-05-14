# 🎮 Pokemon Image Classifier 🔍

An interactive web application that uses deep learning to identify Pokemon from images! Built with TensorFlow and Flask, this project brings the magic of Pokemon recognition to your browser.

## ✨ Features

- 🖼️ Upload Pokemon images through drag-and-drop or click-to-upload
- 🤖 Advanced CNN model for accurate Pokemon recognition
- 📊 Get confidence scores for predictions
- 🎯 See alternative Pokemon matches
- 💫 Beautiful, responsive UI design
- ⚡ Real-time image processing

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **ML/DL**: TensorFlow, OpenCV, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, Scikit-learn

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Pokemon_Classifier.git
cd Pokemon_Classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## 🎯 How to Use

1. Open the web interface in your browser
2. Drag and drop a Pokemon image or click to upload
3. Click "Identify Pokemon" button
4. View the results!
   - Top prediction with confidence score
   - Alternative Pokemon matches
   - Probability percentages

## 🎨 Model Architecture

The classifier uses a deep CNN with:
- Multiple convolutional layers with batch normalization
- MaxPooling layers for feature extraction
- Dropout layers to prevent overfitting
- Dense layers for classification
- Softmax activation for probability distribution

## 📝 Notes

- Supported image formats: PNG, JPG, JPEG
- Maximum file size: 16MB
- For best results, use clear, well-lit Pokemon images
- The model works best with official Pokemon artwork or similar style images

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest enhancements
- Submit pull requests

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Acknowledgments

- Pokemon images and names are property of Nintendo/Game Freak
- Thanks to the TensorFlow and Flask communities
- Special thanks to all contributors and testers

---
Made with ❤️ by [Baasil]