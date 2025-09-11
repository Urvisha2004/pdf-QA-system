from django.contrib import admin
from .models import PDFDocument, QuestionHistory

admin.site.register(PDFDocument)
admin.site.register(QuestionHistory)