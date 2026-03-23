from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, date
import os, json, numpy as np, uuid, pandas as pd, base64
import face_recognition

from database import db, Admin, Student, Attendance, Subject

app = Flask(__name__)

# ---------------- CONFIG ----------------
app.config['SECRET_KEY'] = 'secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads/faces'
app.config['TEMP_FOLDER'] = 'static/uploads/temp'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "student_login"

# ---------------- USER LOADER ----------------
@login_manager.user_loader
def load_user(user_id):
    if session.get("user_type") == "admin":
        return Admin.query.get(int(user_id))
    elif session.get("user_type") == "student":
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
        admin = Admin.query.filter_by(username=request.form['username']).first()

        if admin and check_password_hash(admin.password, request.form['password']):
            session['user_type'] = "admin"
            login_user(admin)
            return redirect(url_for("admin_dashboard"))

        flash("Invalid admin credentials")

    return render_template("admin_login.html")

# ---------------- STUDENT LOGIN ----------------
@app.route('/student-login', methods=['GET','POST'])
def student_login():
    if request.method == "POST":
        student = Student.query.filter_by(student_id=request.form['student_id']).first()

        if student and check_password_hash(student.password, request.form['password']):
            session['user_type'] = "student"
            login_user(student)
            return redirect(url_for("student_dashboard"))

        flash("Invalid student login")

    return render_template("student_login.html")

# ---------------- REGISTER ----------------
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == "POST":

        file = request.files['photo']
        filename = secure_filename(f"{request.form['student_id']}_{uuid.uuid4().hex}.jpg")
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)

        if len(enc) == 0:
            os.remove(path)
            flash("No face detected")
            return redirect(request.url)

        student = Student(
            student_id=request.form['student_id'],
            name=request.form['name'],
            email=request.form['email'],
            password=generate_password_hash(request.form['password']),
            face_encoding=json.dumps(enc[0].tolist()),
            photo_path=filename,
            registered=True
        )

        db.session.add(student)
        db.session.commit()

        flash("Registered successfully")
        return redirect(url_for("student_login"))

    return render_template("register.html")

# ---------------- DASHBOARDS ----------------
@app.route('/admin-dashboard')
@login_required
def admin_dashboard():
    if session.get("user_type") != "admin":
        return redirect(url_for("index"))

    records = Attendance.query.all()
    subjects = Subject.query.all()

    return render_template("admin_dashboard.html", records=records, subjects=subjects)

@app.route('/student-dashboard')
@login_required
def student_dashboard():

    records = Attendance.query.filter_by(
        student_id=current_user.student_id
    ).all()

    # ✅ STRICT JSON SAFE DATA (ONLY DICT)
    attendance_json = []

    for r in records:
        attendance_json.append({
            "subject": str(r.subject),
            "status": str(r.status)
        })

    return render_template(
        "student_dashboard.html",
        attendance=records,          # for table
        attendance_json=attendance_json   # for graph
    )
# ---------------- CAMERA ----------------
@app.route('/camera')
@login_required
def camera_page():
    subjects = Subject.query.all()
    return render_template("camera.html", subjects=subjects)

# ---------------- FACE RECOGNITION ----------------
@app.route('/api/recognize-face', methods=['POST'])
def recognize_face():

    data = request.json
    image_data = data["image"]
    subject = data["subject"]
    hour = data["class_hour"]

    img_bytes = base64.b64decode(image_data.split(",")[1])

    temp_path = os.path.join(app.config['TEMP_FOLDER'], "temp.jpg")
    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    img = face_recognition.load_image_file(temp_path)
    enc = face_recognition.face_encodings(img)

    if len(enc) == 0:
        return jsonify({"error":"No face detected"})

    encoding = enc[0]

    students = Student.query.filter_by(registered=True).all()

    best_match = None
    best_distance = 0.5

    for s in students:
        stored = np.array(json.loads(s.face_encoding))
        dist = face_recognition.face_distance([stored], encoding)[0]

        if dist < best_distance:
            best_distance = dist
            best_match = s

    os.remove(temp_path)

    if best_match:
        today = date.today()

        exists = Attendance.query.filter_by(
            student_id=best_match.student_id,
            subject=subject,
            hour=int(hour),
            date=today
        ).first()

        if exists:
            return jsonify({"message":"Already marked"})

        record = Attendance(
            student_id=best_match.student_id,
            student_name=best_match.name,
            subject=subject,
            hour=int(hour),
            date=today,
            time=datetime.now().time(),
            status="present"
        )

        db.session.add(record)
        db.session.commit()

        return jsonify({
    "student": best_match.name,
    "status": "success"
})
        

    return jsonify({"error":"Not recognized"})

# ---------------- SUBJECT API ----------------
@app.route('/api/subjects')
def subjects_api():
    return jsonify([{"name": s.name} for s in Subject.query.all()])

# ---------------- VIEW ATTENDANCE ----------------
from datetime import datetime

@app.route('/api/attendance/view')
def view_attendance():

    date_filter = request.args.get("date")
    subject_filter = request.args.get("subject")

    print("DATE:", date_filter)       # DEBUG
    print("SUBJECT:", subject_filter) # DEBUG

    query = Attendance.query

    # ✅ DATE FILTER
    if date_filter:
        try:
            date_obj = datetime.strptime(date_filter, "%Y-%m-%d").date()
            query = query.filter(Attendance.date == date_obj)
        except Exception as e:
            print("Date error:", e)

    # ✅ SUBJECT FILTER (ONLY IF NOT EMPTY)
    if subject_filter and subject_filter != "":
       query = query.filter(Attendance.subject == subject_filter)
    records = query.all()

    print("RESULT COUNT:", len(records))  # DEBUG

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
        } for r in records
    ])
# ---------------- EDIT ----------------
@app.route('/api/attendance/edit', methods=['POST'])
def edit_attendance():
    data = request.json
    record = Attendance.query.get(data['id'])

    if record:
        record.status = data['status']
        db.session.commit()
        return jsonify({"success": True})

    return jsonify({"success": False})

# ---------------- DELETE ----------------
@app.route('/api/attendance/delete', methods=['POST'])
def delete_attendance():
    data = request.json
    record = Attendance.query.get(data['id'])

    if record:
        db.session.delete(record)
        db.session.commit()
        return jsonify({"success": True})

    return jsonify({"success": False})

# ---------------- DOWNLOAD ----------------
@app.route('/download-attendance')
@login_required
def download_attendance():

    data = [{
        "Student ID": r.student_id,
        "Name": r.student_name,
        "Subject": r.subject,
        "Date": str(r.date),
        "Time": str(r.time),
        "Hour": r.hour,
        "Status": r.status
    } for r in Attendance.query.all()]

    file = "attendance.xlsx"
    pd.DataFrame(data).to_excel(file, index=False)

    return send_file(file, as_attachment=True)
# ---------------- UPLOAD PAGE ----------------
@app.route('/upload-attendance')
@login_required
def mark_attendance_upload():
    if session.get("user_type") != "admin":
        return redirect(url_for("index"))

    subjects = Subject.query.all()
    return render_template("upload_photo.html", subjects=subjects)

# ---------------- UPLOAD PHOTOS PROCESS ----------------
@app.route('/upload-photos', methods=['POST'])
@login_required
def upload_photos():

    files = request.files.getlist('photos')

    # ✅ GET FROM FRONTEND
    subject = request.form.get("subject")
    hour = request.form.get("hour")

    results = []
    success = 0
    failed = 0

    for file in files:
        try:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['TEMP_FOLDER'], filename)
            file.save(path)

            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)

            if len(enc) == 0:
                results.append({"name": "No face", "status": "failed"})
                failed += 1
                continue

            encoding = enc[0]

            students = Student.query.filter_by(registered=True).all()

            best_match = None
            best_distance = 0.5

            for s in students:
                stored = np.array(json.loads(s.face_encoding))
                dist = face_recognition.face_distance([stored], encoding)[0]

                if dist < best_distance:
                    best_distance = dist
                    best_match = s

            if best_match:

                # ✅ PREVENT DUPLICATE
                exists = Attendance.query.filter_by(
                    student_id=best_match.student_id,
                    subject=subject,
                    hour=int(hour),
                    date=date.today()
                ).first()

                if exists:
                    continue

                record = Attendance(
                    student_id=best_match.student_id,
                    student_name=best_match.name,
                    subject=subject,        # ✅ FIXED
                    hour=int(hour),        # ✅ FIXED
                    date=date.today(),
                    time=datetime.now().time(),
                    status="present"
                )

                db.session.add(record)

                results.append({
                    "name": best_match.name,
                    "status": "success"
                })
                success += 1
            else:
                results.append({
                    "name": "Unknown Student",
                    "status": "failed"
                })
                failed += 1

        except:
            results.append({"name": "Error", "status": "failed"})
            failed += 1

    db.session.commit()

    return jsonify({
        "total": len(files),
        "success": success,
        "failed": failed,
        "results": results
    })
#---------student--------------------------
    
@app.route('/api/students')
def students_api():
    students = Student.query.all()

    return jsonify([
        {
            "student_id": s.student_id,
            "name": s.name
        }
        for s in students
    ])
    
 #---------------- MANUAL MARK ----------------   a
@app.route('/api/manual-mark', methods=['POST'])
@login_required
def manual_mark():

    data = request.json

    student = Student.query.filter_by(student_id=data['student_id']).first()

    if not student:
        return jsonify({"message": "Student not found"})

    # 🔥 PREVENT DUPLICATE
    exists = Attendance.query.filter_by(
        student_id=student.student_id,
        subject=data['subject'],
        hour=int(data['hour']),
        date=date.today()
    ).first()

    if exists:
        return jsonify({"message": "Already marked"})

    record = Attendance(
        student_id=student.student_id,
        student_name=student.name,
        subject=data['subject'],
        hour=int(data['hour']),
        date=date.today(),
        time=datetime.now().time(),
        status="present"
    )

    db.session.add(record)
    db.session.commit()

    return jsonify({"message": f"{student.name} marked successfully"})
# ---------------- LOGOUT ----------------
@app.route('/logout')
@login_required
def logout():
    session.clear()
    logout_user()
    return redirect(url_for("index"))

# ---------------- INIT DB ----------------
def init_db():
    with app.app_context():
        db.create_all()

        if not Admin.query.first():
            db.session.add(Admin(
                username="admin",
                password=generate_password_hash("admin123"),
                email="admin@email.com"
            ))

        if not Subject.query.first():
            subjects = [
                ("Software Testing","ST101"),
                ("Entrepreneurship","ENT101"),
                ("Indian Constitution","IC101"),
                ("Open Elective","OE101")
            ]
            for n,c in subjects:
                db.session.add(Subject(name=n, code=c))

        db.session.commit()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)