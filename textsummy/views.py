from django.shortcuts import render
from django.views.generic import TemplateView,FormView
# from .spacy_summarization import text_summarizer
from .nltk_summarization import nltk_summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration,pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from django.core.files.storage import default_storage
from bs4 import BeautifulSoup
from urllib.request import urlopen
from textsummy.models import *
from django.conf import settings
from .forms import *
import spacy
import PyPDF2
from sumy.summarizers.text_rank import TextRankSummarizer

# Sumy 
def sumy_doc_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

def summy_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    summarizer = TextRankSummarizer()
    summarizer.stop_words = ['in', 'the', 'and', 'but', 'a', 'an']
    summary = summarizer(parser.document, sentences_count=10)
    summary_text = [str(sentence) for sentence in summary]
    return ' '.join(summary_text)
 
def spaacy_summary(input_text, num_sentences=30):
    # Load the Spacy model
    nlp = spacy.load('en_core_web_sm')

    # Process the input text
    doc = nlp(input_text)

    # Get the sentences and their corresponding scores
    sentence_scores = {}
    for sent in doc.sents:
        sentence_scores[sent] = sent.similarity(doc)

    # Sort the sentences by score and get the top `num_sentences`
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Join the top sentences to create the summary
    summary = ' '.join([sent.text for sent in top_sentences])

    return summary
 
# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

# Define a function to generate a summary
def generate_summary(text):
    # Initialize the model and tokenizer
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Generate a summary
    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def lsa(text, n=1):
    # Create a pipeline to extract the latent topics
    pipeline = make_pipeline(
        TfidfVectorizer(),
        TruncatedSVD(n_components=n)
    )

    # Fit the pipeline to the text data and extract the top n topics
    pipeline.fit_transform([text])
    topics = pipeline.named_steps['truncatedsvd'].components_

    # Create the summary by selecting the top sentences based on the most important topics
    sentences = text.split('.')
    sentence_scores = []
    for i in range(len(sentences)):
        sentence_score = 0
        for j in range(n):
            sentence_score += abs(topics[j][i])
        sentence_scores.append((i, sentence_score))
    top_sentences = sorted(sentence_scores, key=lambda item: item[1], reverse=True)[:n]

    # Create the
    summary = ''
    for i in sorted([x[0] for x in top_sentences]):
        summary += sentences[i] + '.'

    return summary

def get_text_from_pdf(pdf_path):
    with open(pdf_path,'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        text = ''
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text()
        return text

class HomePageView(TemplateView):
    template_name = 'home.html'

class TextSummaryPageView(FormView):
    template_name = 'text_summary.html'
    success_url = '/text-summary/'
    form_class = TextSummaryForm

    def form_valid(self, form):
        text = form.cleaned_data['text_summary']
        nltk_summary = nltk_summarizer(text)
        sumy_summary = summy_summary(text)
        spacy_summary = spaacy_summary(text)
        t5_summary   = generate_summary(text)
        lsa_summary  = lsa(text)
        context      = self.get_context_data(form=form, 
                                             spacy_summary=spacy_summary,
                                             nltk_summary=nltk_summary,
                                             t5_summary=t5_summary,
                                             sumy_summary=sumy_summary,
                                             lsa_summary=lsa_summary
                                             )
        return self.render_to_response(context)
    
    
class PdfSummaryPageView(FormView):
    template_name = 'pdf_summary.html'
    success_url = '/pdf-summary/'
    form_class = PDFSummaryForm

    def form_valid(self, form):
        file = form.cleaned_data['pdf_summary']
        file_path = default_storage.save(file.name, file)
        file_url = self.request.scheme + '://' + self.request.get_host() + settings.MEDIA_URL + file_path
        text = get_text_from_pdf(f"media\\{file_path}")
        nltk_summary = nltk_summarizer(text)
        sumy_summary = summy_summary(text)
        spacy_summary = spaacy_summary(text)
        t5_summary   = generate_summary(text)
        lsa_summary  = lsa(text)
        context      = self.get_context_data(form=form, 
                                             spacy_summary=spacy_summary,
                                             nltk_summary=nltk_summary,
                                             t5_summary=t5_summary,
                                             sumy_summary=sumy_summary,
                                             lsa_summary=lsa_summary
                                             )
        return self.render_to_response(context)


class VideoSummaryPageView(TemplateView):
    template_name = 'video_summary.html'

class UrlSummaryPageView(TemplateView):
    template_name = 'url_summary.html'
