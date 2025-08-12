from flask_login import UserMixin
from datetime import datetime

class User(UserMixin):
    def __init__(self, id, email, name, created_at=None):
        self.id = str(id)  # Flask-Login requires string ID
        self.email = email
        self.name = name
        self.created_at = created_at or datetime.utcnow()
    
    def get_id(self):
        return self.id

class Document:
    def __init__(self, id, user_id, filename, file_path, uploaded_at=None, processed=False):
        self.id = id
        self.user_id = user_id
        self.filename = filename
        self.file_path = file_path
        self.uploaded_at = uploaded_at or datetime.utcnow()
        self.processed = processed

class DocumentChunk:
    def __init__(self, id, document_id, content, chunk_index, page_number=None, embedding=None):
        self.id = id
        self.document_id = document_id
        self.content = content
        self.chunk_index = chunk_index
        self.page_number = page_number
        self.embedding = embedding

class ChatMessage:
    def __init__(self, id, user_id, message, response, created_at=None, citations=None):
        self.id = id
        self.user_id = user_id
        self.message = message
        self.response = response
        self.created_at = created_at or datetime.utcnow()
        self.citations = citations or []
