# 📚 Book Recommendation System

This project is a Book Recommendation System developed using collaborative filtering and user-based data. It analyzes user preferences and book ratings to provide personalized book recommendations. Built for educational and demo purposes under the IBM Data Science capstone project.

---

## 🚀 Features

- Recommends books based on user preferences
- Uses collaborative filtering techniques
- Works with real-world datasets (`books.csv`, `users.csv`, `ratings.csv`)
- Lightweight, easy to run locally

---

## 🧩 Tech Stack

- **Python**
- **Pandas** for data processing
- **NumPy**
- **SciKit-learn** (if model-based filtering is included)
- CSV-based datasets

---

## 📁 Files in This Project

```

project ibm/
├── book.py          # Main script for data processing & recommendation
├── books.csv        # Dataset containing book details
├── users.csv        # Dataset containing user info
├── ratings.csv      # Dataset with user-book ratings

````

---

## 🔧 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/book-recommender.git
cd book-recommender
````

2. **Install dependencies:**

```bash
pip install pandas numpy
```

3. **Run the script:**

```bash
python book.py
```

> Make sure `books.csv`, `users.csv`, and `ratings.csv` are in the same directory.

---

## ✅ Use Cases

* Educational recommender system demo
* Personal book suggestion engine
* Basis for further NLP integration or UI development

---

## 📌 Future Improvements

* Add a web UI using Flask or Streamlit
* Integrate NLP for book summary analysis
* Add collaborative + content-based hybrid model

Want me to tailor it for deployment (like with Streamlit or Flask)? Or should I explore the code in `book.py` and summarize what exactly it does?
```
