import os
import re

import pymysql
from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key")

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "image_classification_db")

EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


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
        connection.commit()
    finally:
        connection.close()


def validate_registration_form(full_name, email, password, confirm_password):
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
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, full_name, email FROM users WHERE id = %s",
                    (user_id,),
                )
                return cursor.fetchone()
        finally:
            connection.close()
    except pymysql.MySQLError:
        return None


def login_required(view_function):
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    wrapped_view.__name__ = view_function.__name__
    return wrapped_view


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

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
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
                        return redirect(url_for("dashboard"))
                finally:
                    connection.close()
            except pymysql.MySQLError:
                flash("Unable to connect to MySQL. Please check your database settings.", "error")

    return render_template("login.html", form_data=form_data)


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    if not user:
        session.clear()
        flash("Please log in again.", "error")
        return redirect(url_for("login"))

    return render_template("dashboard.html", user=user)


@app.route("/previous-results")
@login_required
def previous_results():
    user = current_user()
    if not user:
        session.clear()
        flash("Please log in again.", "error")
        return redirect(url_for("login"))

    return render_template("previous_results.html", user=user)


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


if __name__ == "__main__":
    app.run(debug=True)
