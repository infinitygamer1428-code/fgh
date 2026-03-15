from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, date
import os
import json
import numpy as np
import uuid
import pandas as pd

from database import db, Admin, Student, Attendance, Subject
from face_recognition import FaceRecognition

app = Flask(__name__)

# ---------------- CONFIG ----------------

app.config['SECRET_KEY'] = 'secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads/faces'
app.config['TEMP_FOLDER'] = 'static/uploads/temp'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "student_login"

face_recognizer = FaceRecognition()

# ---------------- USER LOADER ----------------

@login_manager.user_loader
def load_user(user_id):

    user_type = session.get("user_type")

    if user_type == "admin":
        return Admin.query.get(int(user_id))

    if user_type == "student":
        return Student.query.get(int(user_id))

    return None


# ---------------- HOME ----------------

@app.route('/')
def index():
    return render_template("index.html")


# ---------------- ADMIN LOGIN ----------------

@app.route('/admin-login', methods=['GET','POST'])
def admin_login():

    if request.method == "POST":

        username = request.form['username']
        password = request.form['password']

        admin = Admin.query.filter_by(username=username).first()

        if admin and check_password_hash(admin.password, password):

            session['user_type'] = "admin"
            login_user(admin)

            return redirect(url_for("admin_dashboard"))

        flash("Invalid admin credentials")

    return render_template("admin_login.html")


# ---------------- STUDENT LOGIN ----------------

@app.route('/student-login', methods=['GET','POST'])
def student_login():

    if request.method == "POST":

        student_id = request.form['student_id']
        password = request.form['password']

        student = Student.query.filter_by(student_id=student_id).first()

        if student and check_password_hash(student.password, password):

            session['user_type'] = "student"
            login_user(student)

            return redirect(url_for("student_dashboard"))

        flash("Invalid student login")

    return render_template("student_login.html")


# ---------------- REGISTER STUDENT ----------------

@app.route('/register', methods=['GET','POST'])
def register():

    if request.method == "POST":

        student_id = request.form['student_id']
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        photo = request.files['photo']

        filename = secure_filename(f"{student_id}_{uuid.uuid4().hex}.jpg")
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        photo.save(path)

        encoding = face_recognizer.get_face_encoding_from_file(path)

        if encoding is None:

            os.remove(path)
            flash("No face detected")
            return redirect(request.url)

        student = Student(
            student_id=student_id,
            name=name,
            email=email,
            password=generate_password_hash(password),
            face_encoding=json.dumps(encoding.tolist()),
            photo_path=filename,
            registered=True
        )

        db.session.add(student)
        db.session.commit()

        flash("Registration successful")

        return redirect(url_for("student_login"))

    return render_template("register.html")


# ---------------- ADMIN DASHBOARD ----------------
@app.route('/admin-dashboard')
@login_required
def admin_dashboard():

    if session.get("user_type") != "admin":
        return redirect(url_for("index"))

    subject_filter = request.args.get("subject")

    query = Attendance.query

    if subject_filter:
        query = query.filter_by(subject=subject_filter)

    records = query.order_by(Attendance.date.desc()).all()

    subjects = Subject.query.all()

    return render_template(
        "admin_dashboard.html",
        records=records,
        subjects=subjects,
        selected_subject=subject_filter
    )

# ---------------- STUDENT DASHBOARD ----------------

@app.route('/student-dashboard')
@login_required
def student_dashboard():

    if session.get("user_type") != "student":
        return redirect(url_for("student_login"))

    attendance = Attendance.query.filter_by(
        student_id=current_user.student_id
    ).all()

    return render_template(
        "student_dashboard.html",
        student=current_user,
        attendance=attendance
    )


# ---------------- CAMERA PAGE ----------------

@app.route('/camera')
@login_required
def camera_page():

    if session.get("user_type") != "admin":
        return redirect(url_for("index"))

    subjects = Subject.query.all()

    return render_template("camera.html", subjects=subjects)


# ---------------- MARK ATTENDANCE UPLOAD ----------------
@app.route('/mark-attendance-upload')
@login_required
def mark_attendance_upload():

    if session.get("user_type") != "admin":
        return redirect(url_for("index"))

    subjects = Subject.query.all()

    return render_template("mark_attendance_upload.html", subjects=subjects)
# ---------------- FACE RECOGNITION ----------------

@app.route('/api/recognize-face', methods=['POST'])
def recognize_face():

    if session.get("user_type") != "admin":
        return jsonify({"error":"Unauthorized"})

    data = request.json

    image = data["image"]
    subject = data["subject"]
    hour = data["class_hour"]

    encoding = face_recognizer.get_face_encoding(image)

    if encoding is None:
        return jsonify({"error":"No face detected"})

    students = Student.query.filter_by(registered=True).all()

    best_match = None
    best_distance = 0.6

    for student in students:

        stored = np.array(json.loads(student.face_encoding))

        distance = face_recognizer.compare_faces(
            encoding,
            stored
        )

        if distance < best_distance:
            best_distance = distance
            best_match = student

    if best_match:

        today = date.today()

        existing = Attendance.query.filter_by(
            student_id=best_match.student_id,
            subject=subject,
            hour=int(hour),
            date=today
        ).first()

        if existing:
            return jsonify({"message":"Already marked"})

        attendance = Attendance(
            student_id=best_match.student_id,
            student_name=best_match.name,
            subject=subject,
            hour=int(hour),
            date=today,
            time=datetime.now().time(),
            status="present"
        )

        db.session.add(attendance)
        db.session.commit()

        return jsonify({
            "message":f"Attendance marked for {best_match.name}",
            "student":best_match.name
        })

    return jsonify({"error":"Student not recognized"})

# ---------------- GET SUBJECTS ----------------

@app.route('/api/subjects')
def get_subjects():

    if session.get("user_type") != "admin":
        return jsonify([])

    subjects = Subject.query.all()

    return jsonify([
        {"id": s.id, "name": s.name, "code": s.code}
        for s in subjects
    ])


# ---------------- VIEW ATTENDANCE ----------------
@app.route('/api/attendance/view')
def view_attendance():

    records = Attendance.query.all()

    return jsonify([
        {
            "id": r.id,
            "student_id": r.student_id,
            "student_name": r.student_name,
            "subject": r.subject,
            "date": str(r.date),
            "time": str(r.time),
            "hour": r.hour,
            "status": r.status
        }
        for r in records
    ])
# ---------------- EDIT ATTENDANCE ----------------

@app.route('/api/attendance/edit', methods=['POST'])
def edit_attendance():

    data = request.json

    record = Attendance.query.get(data['id'])

    if not record:
        return jsonify({"success": False})

    record.status = data['status']

    db.session.commit()

    return jsonify({"success": True})


# ---------------- DELETE ATTENDANCE ----------------

@app.route('/api/attendance/delete', methods=['POST'])
def delete_attendance():

    data = request.json

    record = Attendance.query.get(data['id'])

    if not record:
        return jsonify({"success": False})

    db.session.delete(record)
    db.session.commit()

    return jsonify({"success": True})


# ---------------- DOWNLOAD ATTENDANCE ----------------

@app.route('/download-attendance')
@login_required
def download_attendance():

    records = Attendance.query.all()

    data = []

    for r in records:

        data.append({
            "Student ID": r.student_id,
            "Student Name": r.student_name,
            "Subject": r.subject,
            "Date": str(r.date),
            "Time": str(r.time),
            "Hour": r.hour,
            "Status": r.status
        })

    df = pd.DataFrame(data)

    file_path = "attendance.xlsx"

    df.to_excel(file_path, index=False)

    return send_file(file_path, as_attachment=True)


# ---------------- LOGOUT ----------------

@app.route('/logout')
@login_required
def logout():

    session.clear()
    logout_user()

    return redirect(url_for("index"))


# ---------------- INIT DATABASE ----------------

def init_db():

    with app.app_context():

        db.create_all()

        if not Admin.query.filter_by(username="admin").first():

            admin = Admin(
                username="admin",
                password=generate_password_hash("admin123"),
                email="admin@email.com"
            )

            db.session.add(admin)

        if not Subject.query.first():

            subjects = [
                ("Software Testing","ST101"),
                ("Entrepreneurship","ENT101"),
                ("Indian Constitution","IC101"),
                ("Open Elective","OE101"),
                ("CNE Lab","CNE101"),
                ("SDP Lab","SDP101"),
                ("Software Testing Lab","STL101")
            ]

            for name,code in subjects:
                db.session.add(Subject(name=name, code=code))

        db.session.commit()


# ---------------- MAIN ----------------

if __name__ == "__main__":

    init_db()

    app.run(debug=True)