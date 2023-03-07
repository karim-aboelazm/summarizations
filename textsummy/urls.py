from django.urls import path
from textsummy.views import *

app_name = "textsummy"

urlpatterns = [
    path(""              ,HomePageView.as_view()         ,name="home" ),
    path("text-summary/" ,TextSummaryPageView.as_view()  ,name="text" ),
    path("pdf-summary/"  ,PdfSummaryPageView.as_view()   ,name="pdf"  ),
    path("video-summary/",VideoSummaryPageView.as_view() ,name="video"),
    path("url-summary/"  ,UrlSummaryPageView.as_view()   ,name="url"  ),
]

