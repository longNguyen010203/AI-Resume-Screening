from flask import Flask, request, session
from flask import render_template, redirect, url_for
import os
import pickle
import docx
import PyPDF2
import re
from werkzeug.utils import secure_filename

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Thư mục lưu file tải lên
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "4conruoi" 

# Load mô hình và vectorizer
svc_model = pickle.load(open("models/clf.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
le = pickle.load(open("models/encoder.pkl", "rb"))

# Hàm làm sạch văn bản
def clean_resume(txt):
    clean_text = re.sub(r"http\S+\s", " ", txt)
    clean_text = re.sub(r"RT|cc", " ", clean_text)
    clean_text = re.sub(r"#\S+\s", " ", clean_text)
    clean_text = re.sub(r"@\S+", " ", clean_text)
    clean_text = re.sub(r"[^\w\s]", " ", clean_text)  # Loại bỏ ký tự đặc biệt
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text.strip()

# Hàm trích xuất văn bản từ file
def extract_text_from_file(file_path):
    file_ext = file_path.split(".")[-1].lower()
    
    if file_ext == "pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    elif file_ext == "docx":
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
    
    elif file_ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")
    
    return text

# Hàm dự đoán vị trí công việc phù hợp
def predict_category(resume_text):
    cleaned_text = clean_resume(resume_text)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    prediction = svc_model.predict(vectorized_text)
    predicted_category = le.inverse_transform(prediction)
    return predicted_category[0]

# Route trang chính
@app.route("/")
def index():
    return render_template("index.html")

# Route xử lý upload file
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

            # Xử lý file và dự đoán
            try:
                resume_text = extract_text_from_file(file_path)  # Hàm trích xuất văn bản từ file
                predicted_job = predict_category(resume_text)  # Hàm dự đoán công việc
                session["result"] = f"{predicted_job}"
            except Exception as e:
                session["result"] = f"Lỗi: {str(e)}"

            return redirect(url_for("results"))  # Điều hướng đến trang kết quả

    return render_template("upload.html")

@app.route("/results")
def results():
    result = session.get("result", "Không có kết quả nào.")
    return render_template("results.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
