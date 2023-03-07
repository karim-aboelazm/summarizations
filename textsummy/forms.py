from django import forms

class TextSummaryForm(forms.Form):
    text_summary = forms.CharField(widget=forms.Textarea)
    
class PDFSummaryForm(forms.Form):
    pdf_summary = forms.FileField()
    
