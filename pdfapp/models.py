'''from django.db import models
from django.contrib.auth.models import User

class QuestionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    pdf_name = models.CharField(max_length=255)
    question = models.TextField()
    answer = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.question[:50]}"'''


'''from django.db import models



class PDFDocument(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='pdfs/')

    def __str__(self):
        return self.title

class QuestionHistory(models.Model):
    question = models.TextField()  # User's question
    answer = models.TextField()  # Generated answer 
    pdf_document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"Question: {self.question[:50]}..."  # Truncate for display'''

from django.db import models

class PDFDocument(models.Model):
    file = models.FileField(upload_to="pdfs/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name


class QuestionHistory(models.Model):
    document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, null=True, blank=True)
    question = models.TextField()
    answer = models.TextField()
    asked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q: {self.question[:30]}..."
