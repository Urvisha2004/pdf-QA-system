'''from django import forms
from .models import PDFDocument

class PDFQuestionForm(forms.Form):
    pdf = forms.FileField(label="Upload PDF")
    question = forms.CharField(widget=forms.Textarea(attrs={'rows': 3}), label='Ask a question about the PDF')'''
'''==========================my=================

from django import forms


LANGUAGE_CHOICES = [
    ('en', 'English'),
    ('gu', 'Gujarati'),
    ('hi', 'Hindi'),
    ('mr', 'Marathi'),
    ('ta', 'Tamil'),
    ('te', 'Telugu'),
    ('bn', 'Bengali'),
    ('pa', 'Punjabi'),
    ('ur', 'Urdu'),
    ('ml', 'Malayalam'),
    ('kn', 'Kannada'),
    ('or', 'Odia'),
    ('as', 'Assamese'),
]

class PDFQuestionForm(forms.Form):
    pdf = forms.FileField(label="Upload PDF", required=True)
    question = forms.CharField(widget=forms.Textarea, label="Ask your Question", required=True)
    answer_lang = forms.ChoiceField(choices=LANGUAGE_CHOICES, label="Choose Answer Language", required=True)'''

from django import forms

class PDFUploadForm(forms.Form):
    pdf = forms.FileField(label="Upload PDF")

class QuestionForm(forms.Form):
    question = forms.CharField(label="Ask your question", widget=forms.Textarea(attrs={'rows': 3}))
    answer_lang = forms.ChoiceField(label="Choose Answer Language", choices=[
        ('en', 'English'),
        ('hi', 'Hindi'),
        ('fr', 'French'),
        # Add more as needed
    ])


