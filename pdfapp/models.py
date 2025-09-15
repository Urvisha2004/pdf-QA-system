
from django.db import models

'''class PDFDocument(models.Model):
    file = models.FileField(upload_to="pdfs/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.title or self.file.name


class QuestionHistory(models.Model):
    document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=100, db_index=True)  # Track user session
    question = models.TextField()
    answer = models.TextField()
    answer_lang = models.CharField(max_length=10, default="en")  # Track language
    asked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q: {self.question[:50]} | Lang: {self.answer_lang}"'''

class ChatHistory(models.Model):   # 👈 Add this
    session_id = models.CharField(max_length=100)
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.session_id} - Q: {self.question[:30]}..."