from flask import Flask, render_template
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import os, re, joblib
import docx
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer



app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

knn_model_path = "models/resume_screening_model.pkl"
knn_model = joblib.load(knn_model_path)


def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText.strip()

def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        return extract_text(file_path)
    elif file_path.endswith(".docx") or file_path.endswith(".doc"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format")

def preprocess_resume(file_path, tfidf_model_path):
    # Load TF-IDF vectorizer
    tfidf = joblib.load(tfidf_model_path)
    
    # Extract text from file
    raw_text = extract_text_from_file(file_path)
    
    # Clean the text
    cleaned_text = cleanResume(raw_text)
    
    # Convert text to feature vector
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    
    return vectorized_text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Xử lý CV và dự đoán
            try:
                vectorized_resume = preprocess_resume(file_path, "models/resume_screening_model.pkl")
                prediction = knn_model.predict(vectorized_resume)
                result = f"Dự đoán: {prediction[0]}"
                print(result)
            except Exception as e:
                result = f"Lỗi: {str(e)}"

            # Chuyển hướng đến trang kết quả
            return redirect(url_for("results", result=result, filename=filename))

    return render_template("upload.html")


@app.route("/results")
def results():
    return render_template("results.html")

if __name__ == "__main__":
    app.run(debug=True)
