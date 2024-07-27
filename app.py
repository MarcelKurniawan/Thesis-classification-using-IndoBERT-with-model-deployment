from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re
from nltk.corpus import stopwords
import psycopg2
from psycopg2 import sql
import pyrebase
from functools import wraps
import csv
from io import StringIO

config = {
    'apiKey': 'AIzaSyBle1gLxLcBaPgLY4tPp76_ftxeag_nlwc',
    'authDomain': "classificationindobertproject.firebaseapp.com",
    'projectId': "classificationindobertproject",
    'storageBucket': "classificationindobertproject.appspot.com",
    'messagingSenderId': "935810552144",
    'appId': "1:935810552144:web:62b6449754641da6219c55",
    'databaseURL': "https://classificationindobertproject-default-rtdb.firebaseio.com"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

app = Flask(__name__)
app.secret_key = '1234'

# Define the number of classes
num_labels = 5

# Load the model and tokenizer
model_path = "C:/Users/Marcel/Documents/Skripsi/App/model/IndoBERT_Model.pth"
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased", num_labels=num_labels)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Renaming keys in the state dictionary
new_state_dict = {}
for key, value in state_dict.items():
    if key == "linear.weight":
        new_state_dict["classifier.weight"] = value
    elif key == "linear.bias":
        new_state_dict["classifier.bias"] = value
    else:
        new_state_dict[key] = value

model.load_state_dict(new_state_dict)

tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")

# Define the stop words for preprocessing
stop_words = set(stopwords.words('indonesian'))
class_names = {0: 'DS/AI/IS', 1: 'IOT', 2: 'CS/NT', 3: 'GAT/MT', 4: 'SE/MAT'}

# Preprocessing function
def cleansing(text):
    df_clean = text.lower()
    df_clean = re.sub(r"\d+", "", df_clean)
    df_clean = re.sub(r'[^\w\s]', ' ', df_clean)
    df_clean = re.sub(r'\s+', ' ', df_clean)
    df_clean = re.sub(r'\#\S*', '', df_clean)
    df_clean = ' '.join(word for word in df_clean.split() if word not in stop_words)
    return df_clean

# Prediction function
def predict(text):
    cleansed_text = cleansing(text)
    inputs = tokenizer(cleansed_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()  # Convert logits to probabilities
    prediction = torch.argmax(logits, dim=1).item()
    result = class_names[prediction]
    return result, probabilities


# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect("postgresql://postgres:200403@localhost:5432/thesis_db")
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

# Decorator to check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('authenticate'))
        return f(*args, **kwargs)
    return decorated_function

# Define the route for the home page (predictor)
@app.route('/')
@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate():
    if request.method == 'POST':
        email = request.form['user_email']
        password = request.form['user_pwd']
        try:
            user_info = auth.sign_in_with_email_and_password(email, password)
            account_info = auth.get_account_info(user_info['idToken'])
            if not account_info['users'][0]['emailVerified']:
                verify_message = 'Please verify your email'
                return render_template('authenticate.html', umessage=verify_message)
            session['user'] = email  # Store user email in session
            return render_template('index.html')
        except Exception:
            unsuccessful = 'Please check your credentials'
            return render_template('authenticate.html', umessage=unsuccessful)
    return render_template('authenticate.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict_route():
    text = request.form['text']
    prediction, probabilities = predict(text)
    response = {
        'class': prediction,
        'probabilities': probabilities
    }
    return jsonify(response)


# Define the route for fetching slot data
@app.route('/get_slot', methods=['POST'])
def get_slot():
    supervisor_id = request.form['supervisor_id']

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = sql.SQL("SELECT slot FROM supervisor_data WHERE supervisor_id = %s")
        cur.execute(query, (supervisor_id,))
        slot = cur.fetchone()[0]
        cur.close()
        conn.close()
        return jsonify({'slot': slot})
    except Exception as e:
        print(f"Error fetching slot data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Define the route for saving data
@app.route('/save', methods=['POST'])
def save_data():
    title = request.form['title']
    abstract = request.form['abstract']
    supervisor = request.form['supervisor']
    supervisor_id = request.form.get('supervisor_id')
    topic = request.form.get('topic')

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = sql.SQL("INSERT INTO thesis_data (title, abstract, supervisor, supervisor_id, topic) VALUES (%s, %s, %s, %s, %s)")
        cur.execute(query, (title, abstract, supervisor, supervisor_id, topic))
        
        if supervisor_id:
            update_slot_query = sql.SQL("UPDATE supervisor_data SET slot = slot - 1 WHERE supervisor_id = %s")
            cur.execute(update_slot_query, (supervisor_id,))
        
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Define the route for displaying results
@app.route('/results')
@login_required
def results():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM thesis_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('results.html', rows=rows)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching results."

@app.route('/homepage', methods=['GET'])
@login_required
def homepage():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching supervisor list."

# Define the route for displaying the supervisor list
@app.route('/supervisor_list')
@login_required
def supervisor_list():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM supervisor_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('supervisor_list.html', rows=rows)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching supervisor list."

@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        pwd0 = request.form['user_pwd0']
        pwd1 = request.form['user_pwd1']
        if pwd0 == pwd1:
            try:
                email = request.form['user_email']
                password = request.form['user_pwd1']
                new_user = auth.create_user_with_email_and_password(email, password)
                auth.send_email_verification(new_user['idToken'])
                return render_template('verify_email.html')
            except Exception:
                existing_account = 'This email is already used'
                return render_template('create_account.html', exist_message=existing_account)
        else:
            mismatch_message = 'Passwords do not match'
            return render_template('create_account.html', exist_message=mismatch_message)
    return render_template('create_account.html')

@app.route("/reset_password", methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['user_email']
        auth.send_password_reset_email(email)
        return render_template('authenticate.html')
    return render_template('reset_password.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)  # Remove user from session
    return redirect(url_for('authenticate'))

# Route for generating CSV
@app.route('/generate_csv')
@login_required
def generate_csv():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM thesis_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Create a string buffer to hold CSV data
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)

        # Write header
        writer.writerow(['No', 'Title', 'Abstract', 'Supervisor', 'SpvID', 'Topic'])

        # Write data
        for row in rows:
            writer.writerow(row)

        # Get the CSV string from the buffer
        csv_buffer.seek(0)
        csv_data = csv_buffer.getvalue()

        # Create a response with the CSV data
        response = Response(csv_data, mimetype='text/csv')
        response.headers.set("Content-Disposition", "attachment", filename="thesis_data.csv")
        return response
    except Exception as e:
        print(f"Error generating CSV: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
# Route to generate and download CSV
@app.route('/generate_supervisor_csv')
@login_required
def generate_supervisor_csv():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM supervisor_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Kode Dosen', 'Nama Dosen', 'Slot'])  # Header row
        writer.writerows(rows)
        output.seek(0)

        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=supervisor_list.csv"}
        )
    except Exception as e:
        print(f"Error generating CSV: {e}")
        return "An error occurred while generating the CSV.", 500

if __name__ == '__main__':
    app.run(debug=True)
