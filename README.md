This project is a machine learning–based web application that classifies news articles as Real or Fake.
It is built with Python, Scikit-learn, and Streamlit.

🔹 Features
    1) Interactive Streamlit web app for user-friendly predictions.
    2) Supports multiple ML models:
            - Naive Bayes (MultinomialNB)
            - Support Vector Machine (LinearSVC)
    3) Uses TF-IDF Vectorization for text feature extraction.
    4) Real-time prediction: enter a news headline/text and instantly check if it’s fake or real.
    5) Dataset: Trained on the ISOT Fake News Dataset( True.csv & Fake.csv).

🔹 Tech Stack
Frontend/UI: Streamlit
Backend/ML: Scikit-learn, Python
Data Handling: Pandas, NumPy

🔹 How to Run
# Clone the repository
git clone https://github.com/AkrantDebnath/Fake_News_Detection.git
cd Fake_News_Detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
