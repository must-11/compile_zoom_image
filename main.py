import os

from flask import (Flask, redirect, render_template, request, send_file,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

from run import run
from utils import allwed_file

UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if ("file1" not in request.files) | ("file2" not in request.files):
            flash("ファイルがありません")
            return redirect(request.url)
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        if (file1.filename == "") | (file2.filename == ""):
            flash("ファイルがありません")
            return redirect(request.url)
        if file1 and allwed_file(file1.filename):
            filename1 = secure_filename(file1.filename)
            file1.save(os.path.join(app.config["UPLOAD_FOLDER"], filename1))
        if file2 and allwed_file(file2.filename):
            filename2 = secure_filename(file2.filename)
            file2.save(os.path.join(app.config["UPLOAD_FOLDER"], filename2))
        class args:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename1)
            out_path = "tmp/output.png"
            data_dir = "tmp"
            body_path = "data/body.png"
            background_path = os.path.join(app.config["UPLOAD_FOLDER"], filename2)
        run(args)
        return send_file(args.out_path)
    if request.method == "GET":
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True, port=8888)
