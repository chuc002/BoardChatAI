import os
import logging
from flask import render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from app import app
from services.supabase_service import (
    authenticate_user, create_user, get_user_documents, 
    save_document, delete_document, get_document_by_id
)
from services.pdf_processor import process_pdf
from services.openai_service import chat_with_documents

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        name = request.form['name']
        
        try:
            user = create_user(email, password, name)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Registration failed: {str(e)}', 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            user = authenticate_user(email, password)
            if user:
                login_user(user)
                flash('Logged in successfully!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password.', 'error')
        except Exception as e:
            flash(f'Login failed: {str(e)}', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        documents = get_user_documents(current_user.id)
        return render_template('dashboard.html', documents=documents)
    except Exception as e:
        flash(f'Error loading documents: {str(e)}', 'error')
        return render_template('dashboard.html', documents=[])

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename or "unknown.pdf")
        # Add timestamp to avoid conflicts
        import time
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            
            # Save document record
            document = save_document(current_user.id, file.filename, file_path)
            
            # Process PDF in background (for now, do it synchronously)
            try:
                process_pdf(document.id, file_path)
                return jsonify({
                    'success': True, 
                    'message': 'File uploaded and processed successfully',
                    'document_id': document.id
                })
            except Exception as e:
                logging.error(f"PDF processing failed: {str(e)}")
                return jsonify({
                    'success': True, 
                    'message': 'File uploaded but processing failed. Please try again.',
                    'document_id': document.id
                })
                
        except Exception as e:
            logging.error(f"Upload failed: {str(e)}")
            return jsonify({'success': False, 'message': f'Upload failed: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type. Please upload a PDF.'})

@app.route('/delete_document/<document_id>', methods=['POST'])
@login_required
def delete_document_route(document_id):
    try:
        success = delete_document(document_id, current_user.id)
        if success:
            flash('Document deleted successfully', 'success')
        else:
            flash('Failed to delete document', 'error')
    except Exception as e:
        flash(f'Error deleting document: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/chat/message', methods=['POST'])
@login_required
def chat_message():
    try:
        message = request.form['message']
        if not message.strip():
            return jsonify({'success': False, 'message': 'Message cannot be empty'})
        
        # Get AI response
        response_data = chat_with_documents(current_user.id, message)
        
        return jsonify({
            'success': True,
            'response': response_data['response'],
            'citations': response_data['citations']
        })
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return jsonify({'success': False, 'message': f'Chat error: {str(e)}'})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'message': 'File too large. Maximum size is 16MB.'}), 413
