# 🌿 PlantPedia

**PlantPedia** is a web-based plant identification system that uses a Convolutional Neural Network (CNN) to recognize plant species from uploaded images. The system is designed to provide fast, accurate predictions, along with educational insights on identified plants.

---

## 📁 Project Structure

```
PlantPedia/
├── templates/
│   ├── index.html       # Homepage with image upload
│   ├── result.html      # Displays predicted plant and info
│   └── about.html       # About section for PlantPedia
├── uploads/             # Stores user-uploaded images temporarily
├── app.py               # Flask backend to handle routing, model prediction
├── database.csv         # Plant data referenced in the result view
├── E35_D5_LeakyRelu_0.0005_A94.h5  # Trained CNN model weights
└── requirements.txt     # Python packages needed (you should include this)
```

---

## 🚀 How to Run Locally

### 1. 📦 Install Dependencies

Make sure Python is installed. Then install Flask and other required packages:

```bash
pip install -r requirements.txt
```

### 2. ▶️ Start the Flask Server

```bash
python app.py
```

### 3. 🌐 Open in Browser

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🌱 Features

- 🌿 **Upload Image**: Identify a plant by uploading its image.
- 📊 **Prediction Results**: Displays plant species and related data in table format.
- 🧠 **CNN-Based Model**: Uses a trained `.h5` deep learning model for classification.
- 📖 **About Page**: Highlights the features and benefits of PlantPedia.
- 🎨 **Modern UI**: Responsive, accessible, and clean front-end built with HTML/CSS.

---

## 🔒 Deployment

- This app is intended for **local use** or deployment on **Oracle VM**.
- Not configured for Heroku or public cloud, but can be easily modified.

---

## 👨‍💻 Contributors

- Mani Jain (221648)  
- Mansi Jain (2216049)  
- Shruti Agarwal (2216077)

---

## 📄 License

*This project does not use any license.*

---

## 📸 Sample Use

1. Open the home page (`index.html`)
2. Upload a JPG/PNG image of a plant
3. Get predictions + plant details in `result.html`

---

## 🙌 Acknowledgements

- Dataset preprocessing and training done prior to deployment
- UI inspired by modern design libraries and tools like Unsplash and Google Fonts

---

## ✅ To-Do

- [ ] Add user login/authentication (optional future feature)
- [ ] Enable online deployment
- [ ] Expand model dataset and retrain

---

> _"Empowering people to identify, learn, and explore the green world around them."_
