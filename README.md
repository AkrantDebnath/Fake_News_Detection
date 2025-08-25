This project is a machine learning–based web application that classifies news articles as Real or Fake.
It is built with Python, Scikit-learn, and Streamlit.

🔹 Features

Interactive Streamlit web app for user-friendly predictions.

Supports multiple ML models:

Naive Bayes (MultinomialNB)

Support Vector Machine (LinearSVC)

Uses TF-IDF Vectorization for text feature extraction.

Real-time prediction: enter a news headline/text and instantly check if it’s fake or real.

Dataset: Trained on the LIAR dataset (labeled short statements).

🔹 Tech Stack

Frontend/UI: Streamlit

Backend/ML: Scikit-learn, Python

Data Handling: Pandas, NumPy

🔹 How to Run
# Clone the repository
git clone https://github.com/your-username/Fake_News_Detection.git
cd Fake_News_Detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
