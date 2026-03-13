from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, date
import os
import json
import numpy as np
import uuid

from database import db, Admin, Student, Attendance, Subject
from face_recognition import FaceRecognition

app = Flask(__name__)

# ================= CONFIG =================

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
login_manager.login_view = 'student_login'

face_recognizer = FaceRecognition()

# ================= USER LOADER =================

@login_manager.user_loader
def load_user(user_id):

    user = Student.query.get(user_id)
    if user:
        return user

    user = Admin.query.get(user_id)
    return user


# ================= HOME =================

@app.route('/')
def index():
    return render_template("index.html")


# ================= ADMIN LOGIN =================

@app.route('/admin-login', methods=['GET','POST'])
def admin_login():

    if request.method == "POST":

        username = request.form['username']
        password = request.form['password']

        admin = Admin.query.filter_by(username=username).first()

        if admin and check_password_hash(admin.password, password):

            login_user(admin)
            return redirect(url_for('admin_dashboard'))

        flash("Invalid admin login")

    return render_template("admin_login.html")


# ================= STUDENT LOGIN =================

@app.route('/student-login', methods=['GET','POST'])
def student_login():

    if request.method == "POST":

        student_id = request.form['student_id']
        password = request.form['password']

        student = Student.query.filter_by(student_id=student_id).first()

        if student and check_password_hash(student.password, password):

            login_user(student)
            return redirect(url_for('student_dashboard'))

        flash("Invalid student login")

    return render_template("student_login.html")


# ================= REGISTER =================

@app.route('/register', methods=['GET','POST'])
def register():

    if request.method == "POST":

        student = Student(
            student_id=request.form['student_id'],
            name=request.form['name'],
            email=request.form['email'],
            password=generate_password_hash(request.form['password'])
        )

        db.session.add(student)
        db.session.commit()

        flash("Registration successful")

        return redirect(url_for('student_login'))

    return render_template("register.html")


# ================= ADMIN DASHBOARD =================

@app.route('/admin-dashboard')
@login_required
def admin_dashboard():

    if not isinstance(current_user, Admin):
        return redirect(url_for('index'))

    return render_template("admin_dashboard.html")


# ================= STUDENT DASHBOARD =================

@app.route('/student-dashboard')
@login_required
def student_dashboard():

    if not isinstance(current_user, Student):
        return redirect(url_for('student_login'))

    attendance = Attendance.query.filter_by(
        student_id=current_user.student_id
    ).all()

    return render_template(
        "student_dashboard.html",
        student=current_user,
        attendance=attendance
    )


# ================= CAMERA PAGE =================

@app.route('/camera')
@login_required
def camera_page():

    if not isinstance(current_user, Admin):
        return redirect(url_for('index'))

    subjects = Subject.query.all()

    return render_template("camera.html", subjects=subjects)


# ================= PHOTO UPLOAD (FACE REGISTER) =================

@app.route('/upload-photo', methods=['GET','POST'])
@login_required
def upload_photo():

    if not isinstance(current_user, Student):
        return redirect(url_for('index'))

    if request.method == "POST":

        file = request.files['photo']

        filename = secure_filename(
            f"{current_user.student_id}_{uuid.uuid4().hex}.jpg"
        )

        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(path)

        encoding = face_recognizer.get_face_encoding_from_file(path)

        if encoding is None:

            os.remove(path)
            flash("No face detected")
            return redirect(request.url)

        current_user.face_encoding = json.dumps(encoding.tolist())
        current_user.registered = True
        current_user.photo_path = filename

        db.session.commit()

        flash("Face registered successfully")

        return redirect(url_for('student_dashboard'))

    return render_template("upload_photo.html")


# ================= MARK ATTENDANCE =================

@app.route('/mark-attendance-upload', methods=['GET','POST'])
@login_required
def mark_attendance_upload():

    if not isinstance(current_user, Admin):
        return redirect(url_for('index'))

    subjects = Subject.query.all()

    if request.method == "POST":

        file = request.files['photo']
        subject = request.form['subject']
        hour = request.form['class_hour']

        temp_file = os.path.join(
            app.config['TEMP_FOLDER'],
            secure_filename(f"temp_{uuid.uuid4().hex}.jpg")
        )

        file.save(temp_file)

        encoding = face_recognizer.get_face_encoding_from_file(temp_file)

        os.remove(temp_file)

        if encoding is None:
            flash("No face detected")
            return redirect(request.url)

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

            flash(f"Attendance marked for {best_match.name}")

        else:

            flash("Student not recognized")

    return render_template(
        "mark_attendance_upload.html",
        subjects=subjects
    )


# ================= API SUBJECTS =================

@app.route('/api/subjects')
@login_required
def get_subjects():

    subjects = Subject.query.all()

    return jsonify([
        {"id": s.id, "name": s.name, "code": s.code}
        for s in subjects
    ])


# ================= API ATTENDANCE =================

@app.route('/api/attendance/view')
@login_required
def view_attendance():

    records = Attendance.query.all()

    return jsonify([{
        "id": r.id,
        "student_id": r.student_id,
        "student_name": r.student_name,
        "subject": r.subject,
        "date": str(r.date),
        "time": str(r.time),
        "hour": r.hour,
        "status": r.status
    } for r in records])


# ================= LOGOUT =================

@app.route('/logout')
@login_required
def logout():

    logout_user()

    return redirect(url_for('index'))


# ================= INIT DATABASE =================

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
                ("Mathematics","MATH101"),
                ("Physics","PHY101"),
                ("Computer Science","CS101")
            ]

            for name,code in subjects:

                db.session.add(
                    Subject(name=name,code=code)
                )

        db.session.commit()


# ================= MAIN =================

if __name__ == "__main__":

    init_db()

    app.run(debug=True)