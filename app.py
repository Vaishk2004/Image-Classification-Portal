import json
import os
import re
import uuid
from datetime import datetime

import pymysql
from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key")

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "image_classification_db")

EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads", "images")
MODEL_PATH = os.path.join(app.root_path, "models", "cnn_image_classifier.keras")
CLASS_NAMES_PATH = os.path.join(app.root_path, "models", "class_names.json")
CLASS_DISPLAY_NAMES = {
    "animal": "Animal",
    "human": "Human",
    "nonLiving": "Non-Living Thing",
}
loaded_model = None


def get_mysql_connection(use_database=True):
    connection_options = {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "cursorclass": pymysql.cursors.DictCursor,
    }

    if use_database:
        connection_options["database"] = MYSQL_DATABASE

    return pymysql.connect(**connection_options)


def initialize_database():
    connection = get_mysql_connection(use_database=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE `{MYSQL_DATABASE}`")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    full_name VARCHAR(120) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS classification_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    stored_filename VARCHAR(255) NOT NULL,
                    image_path VARCHAR(500) NOT NULL,
                    predicted_category VARCHAR(80) NOT NULL,
                    confidence_score DECIMAL(5, 2) NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
        connection.commit()
    finally:
        connection.close()


def validate_registration_form(full_name, email, password, confirm_password): #validates user input before saving
    errors = []

    if not full_name:
        errors.append("Full name is required.")

    if not email:
        errors.append("Email address is required.")
    elif not EMAIL_PATTERN.match(email):
        errors.append("Please enter a valid email address.")

    if not password:
        errors.append("Password is required.")
    elif len(password) < 6:
        errors.append("Password must be at least 6 characters long.")

    if not confirm_password:
        errors.append("Confirm password is required.")
    elif password != confirm_password:
        errors.append("Password and confirm password must match.")

    return errors


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None

    try:
        initialize_database()
        connection = get_mysql_connection()
        try:
            with connection.cursor() as cursor: #cursor is used to execute SQL queries
                # Search user with id
                cursor.execute(
                    "SELECT id, full_name, email FROM users WHERE id = %s",
                    (user_id,),
                )
                return cursor.fetchone() #returns the first matching row
        finally:
            connection.close()
    except pymysql.MySQLError:
        return None


#Create Decorator
def login_required(view_function):
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for("login", next=request.path))

        return view_function(*args, **kwargs)

    wrapped_view.__name__ = view_function.__name__
    return wrapped_view


def allowed_image_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        return []

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as class_file:
        return json.load(class_file)


def classify_image(image_path):
    global loaded_model

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            "CNN model not found. Please train the model first using train_model.py."
        )

    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image

    if loaded_model is None:
        loaded_model = load_model(MODEL_PATH)

    class_names = load_class_names()
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = loaded_model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    raw_label = class_names[predicted_index]
    confidence = round(float(predictions[predicted_index]) * 100, 2)

    return CLASS_DISPLAY_NAMES.get(raw_label, raw_label), confidence


def preload_cnn_model():
    global loaded_model

    if not os.path.exists(MODEL_PATH):
        return

    import numpy as np
    from tensorflow.keras.models import load_model

    loaded_model = load_model(MODEL_PATH)
    loaded_model.predict(np.zeros((1, 128, 128, 3)), verbose=0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    form_data = {
        "full_name": "",
        "email": "",
    }

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        form_data["full_name"] = full_name
        form_data["email"] = email

        errors = validate_registration_form(full_name, email, password, confirm_password)

        if not errors:
            try:
                initialize_database()
                connection = get_mysql_connection()
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                        existing_user = cursor.fetchone()

                        if existing_user:
                            errors.append("This email address is already registered.")
                        else:
                            password_hash = generate_password_hash(password)
                            cursor.execute(
                                """
                                INSERT INTO users (full_name, email, password_hash)
                                VALUES (%s, %s, %s)
                                """,
                                (full_name, email, password_hash),
                            )
                            user_id = cursor.lastrowid
                            connection.commit()
                            session.clear()
                            session["user_id"] = user_id
                            session["user_name"] = full_name
                            flash("Registration successful. Welcome to your dashboard.", "success")
                            return redirect(url_for("dashboard"))
                finally:
                    connection.close()
            except pymysql.MySQLError:
                errors.append(
                    "Unable to connect to MySQL. Please check your database settings."
                )

        for error in errors:
            flash(error, "error")

    return render_template("register.html", form_data=form_data)


@app.route("/login", methods=["GET", "POST"])
def login():
    form_data = {"email": ""}
    next_page = request.args.get("next") or url_for("dashboard")

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        next_page = request.form.get("next") or url_for("dashboard")
        form_data["email"] = email

        if not email or not password:
            flash("Email address and password are required.", "error")
        else:
            try:
                initialize_database()
                connection = get_mysql_connection()
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT id, full_name, email, password_hash FROM users WHERE email = %s",
                            (email,),
                        )
                        user = cursor.fetchone()

                    if not user or not check_password_hash(user["password_hash"], password):
                        flash("Invalid email address or password.", "error")
                    else:
                        session.clear()
                        session["user_id"] = user["id"]
                        session["user_name"] = user["full_name"]
                        flash("Login successful.", "success")
                        if not next_page.startswith("/"):
                            next_page = url_for("dashboard")
                        return redirect(next_page)
                finally:
                    connection.close()
            except pymysql.MySQLError:
                flash("Unable to connect to MySQL. Please check your database settings.", "error")

    return render_template("login.html", form_data=form_data, next_page=next_page)


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    user = current_user()
    if not user:
        session.clear()
        flash("Please log in again.", "error")
        return redirect(url_for("login"))

    result = None

    if request.method == "POST":
        uploaded_file = request.files.get("image_file")

        if not uploaded_file or uploaded_file.filename == "":
            flash("Please choose an image file to upload.", "error")
        elif not allowed_image_file(uploaded_file.filename):
            flash("Only JPG, JPEG, and PNG image files are supported.", "error")
        else:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            original_filename = secure_filename(uploaded_file.filename)
            extension = original_filename.rsplit(".", 1)[1].lower()
            stored_filename = f"{uuid.uuid4().hex}.{extension}"
            saved_path = os.path.join(UPLOAD_FOLDER, stored_filename)
            uploaded_file.save(saved_path)

            try:
                predicted_category, confidence_score = classify_image(saved_path)
                relative_path = f"uploads/images/{stored_filename}"

                initialize_database()
                connection = get_mysql_connection()
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            INSERT INTO classification_results (
                                user_id, original_filename, stored_filename,
                                image_path, predicted_category, confidence_score
                            )
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                user["id"],
                                original_filename,
                                stored_filename,
                                relative_path,
                                predicted_category,
                                confidence_score,
                            ),
                        )
                    connection.commit()
                finally:
                    connection.close()

                result = {
                    "image_url": url_for("uploaded_image", filename=stored_filename),
                    "predicted_category": predicted_category,
                    "confidence_score": confidence_score,
                    "uploaded_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                }
                flash("Image classified successfully.", "success")
            except FileNotFoundError as error:
                flash(str(error), "error")
            except pymysql.MySQLError:
                flash("Unable to save the result in MySQL. Please check your database settings.", "error")
            except Exception as error:
                flash(f"Image classification failed: {error}", "error")

    return render_template("dashboard.html", user=user, result=result)


@app.route("/previous-results")
@login_required
def previous_results():
    user = current_user()
    if not user:
        session.clear()
        flash("Please log in again.", "error")
        return redirect(url_for("login"))

    results = []
    try:
        initialize_database()
        connection = get_mysql_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, stored_filename, predicted_category, confidence_score, uploaded_at
                    FROM classification_results
                    WHERE user_id = %s
                    ORDER BY uploaded_at DESC
                    """,
                    (user["id"],),
                )
                results = cursor.fetchall()
        finally:
            connection.close()
    except pymysql.MySQLError:
        flash("Unable to load previous results. Please check your database settings.", "error")

    return render_template("previous_results.html", user=user, results=results)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out securely.", "success")
    return redirect(url_for("login"))


@app.route("/assets/<path:filename>")
def asset_file(filename):
    if filename not in {"OIP.jpeg", "style.css"}:
        abort(404)

    return send_from_directory(app.root_path, filename)


@app.route("/uploads/images/<path:filename>")
@login_required
def uploaded_image(filename):
    user = current_user()
    if not user:
        session.clear()
        flash("Please log in again.", "error")
        return redirect(url_for("login"))

    try:
        initialize_database()
        connection = get_mysql_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id FROM classification_results
                    WHERE user_id = %s AND stored_filename = %s
                    """,
                    (user["id"], filename),
                )
                result = cursor.fetchone()
        finally:
            connection.close()
    except pymysql.MySQLError:
        abort(404)

    if not result:
        abort(404)

    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    preload_cnn_model()
    app.run(debug=False, use_reloader=False)
