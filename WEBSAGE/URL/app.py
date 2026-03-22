import streamlit as st
from streamlit import session_state
import json
import os
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from collections import deque
import time
import pandas as pd
import matplotlib.pyplot as plt
import re
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "language" not in st.session_state:
    st.session_state.language = "English"
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = []
if "product_recommendations" not in st.session_state:
    st.session_state.product_recommendations = {}

# Language translations including Indian languages
translations = {
    "English": {
        "title": "Web Analysis Tool",
        "welcome": "Welcome to the Web Analysis Tool. Use the sidebar to navigate.",
        "navigation": "Navigation",
        "home": "Home",
        "analysis": "Analysis",
        "comparison": "Comparison",
        "dashboard": "Dashboard",
        "auth": "Authentication",
        "login_btn": "Login",
        "signup_btn": "Sign Up",
        "login_title": "Login",
        "email": "Email",
        "password": "Password",
        "submit_login": "Login",
        "signup_title": "Sign Up",
        "submit_signup": "Sign Up",
        "fill_all_fields": "Please fill in all fields",
        "email_exists": "Email already exists",
        "account_created": "Account created successfully! Please login.",
        "login_success": "Login successful!",
        "invalid_credentials": "Invalid email or password",
        "enter_url": "Enter URL to analyze:",
        "start_scraping": "Start Scraping",
        "scraping_analyzing": "Scraping and analyzing content...",
        "scraping_failed": "Failed to scrape content from the provided URL. Please try again.",
        "no_content": "No content to analyze. Please try a different URL.",
        "analysis_error": "An error occurred during analysis. Please try again.",
        "download_results": "Download Results",
        "download_html": "Download as HTML",
        "enter_first_url": "Enter first URL:",
        "enter_second_url": "Enter second URL:",
        "comparison_question": "Enter comparison question (optional):",
        "question_placeholder": "e.g., 'Compare product offerings' or 'List all products'",
        "start_comparison": "Start Comparison",
        "scraping_first": "Failed to scrape content from {}. Please try again.",
        "scraping_second": "Failed to scrape content from {}. Please try again.",
        "comparison_error": "Error during comparison: {}",
        "enter_both_urls": "Please enter both URLs to compare.",
        "user_dashboard": "User Dashboard",
        "welcome_user": "Welcome, {}!",
        "logout": "Logout",
        "language_select": "Select Language",
        "content_analysis": "Content Analysis",
        "summary": "Summary",
        "products": "Products",
        "price_distribution": "Price Distribution",
        "categories": "Categories",
        "category_distribution": "Category Distribution",
        "brands": "Brands",
        "brand_distribution": "Brand Distribution",
        "materials": "Materials",
        "material_distribution": "Material Distribution",
        "colors": "Colors",
        "color_distribution": "Color Distribution",
        "sizes": "Sizes",
        "size_distribution": "Size Distribution",
        "statistics": "Statistics",
        "content_metrics": "Content Metrics",
        "text_length": "Text Length",
        "total_characters": "Total Characters",
        "website_comparison": "Website Comparison",
        "comparison_analysis": "Comparison Analysis",
        "statistics_comparison": "Statistics Comparison",
        "first_website": "First Website",
        "second_website": "Second Website",
        "metric": "Metric",
        "products_comparison": "Products Comparison",
        "price_comparison": "Price Distribution Comparison",
        "no_products": "No products found",
        "pages": "Pages",
        "images": "Images",
        "tables": "Tables",
        "lists": "Lists",
        "metrics": "Metrics",
        "sentiment_analysis": "Sentiment Analysis",
        "reviews": "Reviews",
        "sentiment_distribution": "Sentiment Distribution",
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "average_rating": "Average Rating",
        "review_insights": "Review Insights",
        "similar_projects": "Similar Projects",
        "recommended_websites": "Recommended Websites",
        "similarity_score": "Similarity Score",
        "content_similarity": "Content Similarity",
        "product_recommendations": "Product Recommendations",
        "view_similar": "View Similar Products",
        "based_on_analysis": "Based on your analysis history",
        "trending_products": "Trending Products",
        "top_rated": "Top Rated",
        "most_reviewed": "Most Reviewed",
        "project_suggestions": "Project Suggestions",
        "review_sentiment": "Review Sentiment",
        "product_alternatives": "Product Alternatives",
        "if_you_liked": "If you liked",
        "here_are_alternatives": "here are some alternatives",
        "sentiment_score": "Sentiment Score",
        "overall_sentiment": "Overall Sentiment",
        "review_analysis": "Review Analysis",
        "recommendation_engine": "Recommendation Engine",
        "similar_products": "Similar Products",
        "better_alternatives": "Better Alternatives",
        "price_range": "Price Range",
        "rating_comparison": "Rating Comparison",
        "feature_comparison": "Feature Comparison",
    },
    "Hindi": {
        "title": "वेब विश्लेषण उपकरण",
        "welcome": "वेब विश्लेषण उपकरण में आपका स्वागत है। नेविगेट करने के लिए साइडबार का उपयोग करें।",
        "navigation": "नेविगेशन",
        "home": "होम",
        "analysis": "विश्लेषण",
        "comparison": "तुलना",
        "dashboard": "डैशबोर्ड",
        "auth": "प्रमाणीकरण",
        "login_btn": "लॉगिन",
        "signup_btn": "साइन अप",
        "login_title": "लॉगिन",
        "email": "ईमेल",
        "password": "पासवर्ड",
        "submit_login": "लॉगिन",
        "signup_title": "साइन अप",
        "submit_signup": "साइन अप",
        "fill_all_fields": "कृपया सभी फ़ील्ड भरें",
        "email_exists": "ईमेल पहले से मौजूद है",
        "account_created": "खाता सफलतापूर्वक बनाया गया! कृपया लॉगिन करें।",
        "login_success": "लॉगिन सफल!",
        "invalid_credentials": "अमान्य ईमेल या पासवर्ड",
        "enter_url": "विश्लेषण के लिए URL दर्ज करें:",
        "start_scraping": "स्क्रैपिंग शुरू करें",
        "scraping_analyzing": "सामग्री स्क्रैप और विश्लेषण किया जा रहा है...",
        "scraping_failed": "प्रदान किए गए URL से सामग्री स्क्रैप करने में विफल। कृपया पुनः प्रयास करें।",
        "no_content": "विश्लेषण के लिए कोई सामग्री नहीं। कृपया एक अलग URL आज़माएँ।",
        "analysis_error": "विश्लेषण के दौरान एक त्रुटि हुई। कृपया पुनः प्रयास करें।",
        "download_results": "परिणाम डाउनलोड करें",
        "download_html": "HTML के रूप में डाउनलोड करें",
        "enter_first_url": "पहला URL दर्ज करें:",
        "enter_second_url": "दूसरा URL दर्ज करें:",
        "comparison_question": "तुलना प्रश्न दर्ज करें (वैकल्पिक):",
        "question_placeholder": "जैसे, 'उत्पाद offerings की तुलना करें' या 'सभी उत्पादों की सूची बनाएं'",
        "start_comparison": "तुलना शुरू करें",
        "scraping_first": "{} से सामग्री स्क्रैप करने में विफल। कृपया पुनः प्रयास करें।",
        "scraping_second": "{} से सामग्री स्क्रैप करने में विफल। कृपया पुनः प्रयास करें।",
        "comparison_error": "तुलना के दौरान त्रुटि: {}",
        "enter_both_urls": "कृपया तुलना करने के लिए दोनों URL दर्ज करें।",
        "user_dashboard": "उपयोगकर्ता डैशबोर्ड",
        "welcome_user": "आपका स्वागत है, {}!",
        "logout": "लॉगआउट",
        "language_select": "भाषा चुनें",
        "content_analysis": "सामग्री विश्लेषण",
        "summary": "सारांश",
        "products": "उत्पाद",
        "price_distribution": "मूल्य वितरण",
        "categories": "श्रेणियाँ",
        "category_distribution": "श्रेणी वितरण",
        "brands": "ब्रांड",
        "brand_distribution": "ब्रांड वितरण",
        "materials": "सामग्री",
        "material_distribution": "सामग्री वितरण",
        "colors": "रंग",
        "color_distribution": "रंग वितरण",
        "sizes": "आकार",
        "size_distribution": "आकार वितरण",
        "statistics": "आँकड़े",
        "content_metrics": "सामग्री मेट्रिक्स",
        "text_length": "टेक्स्ट लंबाई",
        "total_characters": "कुल वर्ण",
        "website_comparison": "वेबसाइट तुलना",
        "comparison_analysis": "तुलना विश्लेषण",
        "statistics_comparison": "आँकड़े तुलना",
        "first_website": "पहली वेबसाइट",
        "second_website": "दूसरी वेबसाइट",
        "metric": "मेट्रिक",
        "products_comparison": "उत्पाद तुलना",
        "price_comparison": "मूल्य वितरण तुलना",
        "no_products": "कोई उत्पाद नहीं मिला",
        "pages": "पृष्ठ",
        "images": "छवियाँ",
        "tables": "टेबल",
        "lists": "सूचियाँ",
        "metrics": "मेट्रिक्स",
        "sentiment_analysis": "भावना विश्लेषण",
        "reviews": "समीक्षाएं",
        "sentiment_distribution": "भावना वितरण",
        "positive": "सकारात्मक",
        "negative": "नकारात्मक",
        "neutral": "तटस्थ",
        "average_rating": "औसत रेटिंग",
        "review_insights": "समीक्षा अंतर्दृष्टि",
        "similar_projects": "समान परियोजनाएं",
        "recommended_websites": "अनुशंसित वेबसाइटें",
        "similarity_score": "समानता स्कोर",
        "content_similarity": "सामग्री समानता",
        "product_recommendations": "उत्पाद अनुशंसाएं",
        "view_similar": "समान उत्पाद देखें",
        "based_on_analysis": "आपके विश्लेषण इतिहास के आधार पर",
        "trending_products": "ट्रेंडिंग उत्पाद",
        "top_rated": "शीर्ष रेटेड",
        "most_reviewed": "सबसे अधिक समीक्षित",
        "project_suggestions": "परियोजना सुझाव",
        "review_sentiment": "समीक्षा भावना",
        "product_alternatives": "उत्पाद विकल्प",
        "if_you_liked": "अगर आपको पसंद आया",
        "here_are_alternatives": "यहाँ कुछ विकल्प हैं",
        "sentiment_score": "भावना स्कोर",
        "overall_sentiment": "कुल भावना",
        "review_analysis": "समीक्षा विश्लेषण",
        "recommendation_engine": "सिफारिश इंजन",
        "similar_products": "समान उत्पाद",
        "better_alternatives": "बेहतर विकल्प",
        "price_range": "मूल्य सीमा",
        "rating_comparison": "रेटिंग तुलना",
        "feature_comparison": "विशेषता तुलना",
    },
    "Tamil": {
        "title": "வலை பகுப்பாய்வு கருவி",
        "welcome": "வலை பகுப்பாய்வு கருவிக்கு வரவேற்கிறோம். செல்லவும் பக்கப்பட்டையைப் பயன்படுத்தவும்.",
        "navigation": "வழிசெலுத்தல்",
        "home": "முகப்பு",
        "analysis": "பகுப்பாய்வு",
        "comparison": "ஒப்பீடு",
        "dashboard": "டாஷ்போர்டு",
        "auth": "அங்கீகாரம்",
        "login_btn": "உள்நுழைய",
        "signup_btn": "பதிவு செய்ய",
        "login_title": "உள்நுழைய",
        "email": "மின்னஞ்சல்",
        "password": "கடவுச்சொல்",
        "submit_login": "உள்நுழைய",
        "signup_title": "பதிவு செய்ய",
        "submit_signup": "பதிவு செய்ய",
        "fill_all_fields": "தயவு செய்து அனைத்து புலங்களையும் நிரப்பவும்",
        "email_exists": "மின்னஞ்சல் ஏற்கனவே உள்ளது",
        "account_created": "கணக்கு வெற்றிகரமாக உருவாக்கப்பட்டது! தயவு செய்து உள்நுழையவும்.",
        "login_success": "உள்நுழைவு வெற்றிகரமானது!",
        "invalid_credentials": "தவறான மின்னஞ்சல் அல்லது கடவுச்சொல்",
        "enter_url": "பகுப்பாய்வு செய்ய URL ஐ உள்ளிடவும்:",
        "start_scraping": "ஸ்கிராப்பிங் தொடங்கவும்",
        "scraping_analyzing": "உள்ளடக்கத்தை ஸ்கிராப் மற்றும் பகுப்பாய்வு செய்யப்படுகிறது...",
        "scraping_failed": "வழங்கப்பட்ட URL இலிருந்து உள்ளடக்கத்தை ஸ்கிராப் செய்ய முடியவில்லை. தயவு செய்து மீண்டும் முயற்சிக்கவும்.",
        "no_content": "பகுப்பாய்வு செய்ய எந்த உள்ளடக்கமும் இல்லை. தயவு செய்து வேறு URL ஐ முயற்சிக்கவும்.",
        "analysis_error": "பகுப்பாய்வின் போது பிழை ஏற்பட்டது. தயவு செய்து மீண்டும் முயற்சிக்கவும்.",
        "download_results": "முடிவுகளை பதிவிறக்கவும்",
        "download_html": "HTML ஆக பதிவிறக்கவும்",
        "enter_first_url": "முதல் URL ஐ உள்ளிடவும்:",
        "enter_second_url": "இரண்டாவது URL ஐ உள்ளிடவும்:",
        "comparison_question": "ஒப்பீட்டு கேள்வியை உள்ளிடவும் (விருப்பத்தேர்வு):",
        "question_placeholder": "எ.கா., 'தயாரிப்பு offeringsஐ ஒப்பிடுக' அல்லது 'அனைத்து தயாரிப்புகளையும் பட்டியலிடுக'",
        "start_comparison": "ஒப்பீடு தொடங்கவும்",
        "scraping_first": "{} இலிருந்து உள்ளடக்கத்தை ஸ்கிராப் செய்ய முடியவில்லை. தயவு செய்து மீண்டும் முயற்சிக்கவும்.",
        "scraping_second": "{} இலிருந்து உள்ளடக்கத்தை ஸ்கிராப் செய்ய முடியவில்லை. தயவு செய்து மீண்டும் முயற்சிக்கவும்.",
        "comparison_error": "ஒப்பீட்டின் போது பிழை: {}",
        "enter_both_urls": "தயவு செய்து ஒப்பிடுவதற்கு இரண்டு URL களையும் உள்ளிடவும்.",
        "user_dashboard": "பயனர் டாஷ்போர்டு",
        "welcome_user": "வரவேற்கிறோம், {}!",
        "logout": "வெளியேறு",
        "language_select": "மொழியைத் தேர்ந்தெடுக்கவும்",
        "content_analysis": "உள்ளடக்க பகுப்பாய்வு",
        "summary": "சுருக்கம்",
        "products": "தயாரிப்புகள்",
        "price_distribution": "விலை விநியோகம்",
        "categories": "வகைகள்",
        "category_distribution": "வகை விநியோகம்",
        "brands": "பிராண்டுகள்",
        "brand_distribution": "பிராண்டு விநியோகம்",
        "materials": "பொருட்கள்",
        "material_distribution": "பொருள் விநியோகம்",
        "colors": "நிறங்கள்",
        "color_distribution": "நிறம் விநியோகம்",
        "sizes": "அளவுகள்",
        "size_distribution": "அளவு விநியோகம்",
        "statistics": "புள்ளிவிவரங்கள்",
        "content_metrics": "உள்ளடக்க அளவீடுகள்",
        "text_length": "உரை நீளம்",
        "total_characters": "மொத்த எழுத்துகள்",
        "website_comparison": "வலைத்தள ஒப்பீடு",
        "comparison_analysis": "ஒப்பீட்டு பகுப்பாய்வு",
        "statistics_comparison": "புள்ளிவிவரங்கள் ஒப்பீடு",
        "first_website": "முதல் வலைத்தளம்",
        "second_website": "இரண்டாவது வலைத்தளம்",
        "metric": "அளவீடு",
        "products_comparison": "தயாரிப்புகள் ஒப்பீடு",
        "price_comparison": "விலை விநியோகம் ஒப்பீடு",
        "no_products": "தயாரிப்புகள் எதுவும் கிடைக்கவில்லை",
        "pages": "பக்கங்கள்",
        "images": "படங்கள்",
        "tables": "அட்டவணைகள்",
        "lists": "பட்டியல்கள்",
        "metrics": "அளவீடுகள்",
        "sentiment_analysis": "உணர்வு பகுப்பாய்வு",
        "reviews": "விமர்சனங்கள்",
        "sentiment_distribution": "உணர்வு விநியோகம்",
        "positive": "நேர்மறை",
        "negative": "எதிர்மறை",
        "neutral": "நடுநிலை",
        "average_rating": "சராசரி மதிப்பீடு",
        "review_insights": "விமர்சன நுண்ணறிவுகள்",
        "similar_projects": "ஒத்த திட்டங்கள்",
        "recommended_websites": "பரிந்துரைக்கப்பட்ட வலைத்தளங்கள்",
        "similarity_score": "ஒற்றுமை மதிப்பெண்",
        "content_similarity": "உள்ளடக்க ஒற்றுமை",
        "product_recommendations": "தயாரிப்பு பரிந்துரைகள்",
        "view_similar": "ஒத்த தயாரிப்புகளைக் காண்க",
        "based_on_analysis": "உங்கள் பகுப்பாய்வு வரலாற்றின் அடிப்படையில்",
        "trending_products": "பிரபலமான தயாரிப்புகள்",
        "top_rated": "மிக உயர்ந்த மதிப்பீடு",
        "most_reviewed": "அதிகம் விமர்சிக்கப்பட்டது",
        "project_suggestions": "திட்ட பரிந்துரைகள்",
        "review_sentiment": "விமர்சன உணர்வு",
        "product_alternatives": "தயாரிப்பு மாற்றுகள்",
        "if_you_liked": "நீங்கள் விரும்பியிருந்தால்",
        "here_are_alternatives": "இங்கே சில மாற்றுகள் உள்ளன",
        "sentiment_score": "உணர்வு மதிப்பெண்",
        "overall_sentiment": "மொத்த உணர்வு",
        "review_analysis": "விமர்சன பகுப்பாய்வு",
        "recommendation_engine": "பரிந்துரை இயந்திரம்",
        "similar_products": "ஒத்த தயாரிப்புகள்",
        "better_alternatives": "சிறந்த மாற்றுகள்",
        "price_range": "விலை வரம்பு",
        "rating_comparison": "மதிப்பீடு ஒப்பீடு",
        "feature_comparison": "அம்சம் ஒப்பீடு",
    },
    "Bengali": {
        "title": "ওয়েব বিশ্লেষণ টুল",
        "welcome": "ওয়েব বিশ্লেষণ টুলে আপনাকে স্বাগতম। নেভিগেট করতে সাইডবার ব্যবহার করুন।",
        "navigation": "নেভিগেশন",
        "home": "হোম",
        "analysis": "বিশ্লেষণ",
        "comparison": "তুলনা",
        "dashboard": "ড্যাশবোর্ড",
        "auth": "প্রমাণীকরণ",
        "login_btn": "লগইন",
        "signup_btn": "সাইন আপ",
        "login_title": "লগইন",
        "email": "ইমেল",
        "password": "পাসওয়ার্ড",
        "submit_login": "লগইন",
        "signup_title": "সাইন আプ",
        "submit_signup": "সাইন আপ",
        "fill_all_fields": "দয়া করে সব fields পূরণ করুন",
        "email_exists": "ইমেল ইতিমধ্যেই存在",
        "account_created": "অ্যাকাউন্ট সফলভাবে তৈরি হয়েছে! দয়া করে লগইন করুন।",
        "login_success": "লগইন সফল!",
        "invalid_credentials": "অবৈধ ইমেল বা পাসওয়ার্ড",
        "enter_url": "বিশ্লেষণের জন্য URL লিখুন:",
        "start_scraping": "স্ক্র্যাপিং শুরু করুন",
        "scraping_analyzing": "কন্টেন্ট স্ক্র্যাপ এবং বিশ্লেষণ করা হচ্ছে...",
        "scraping_failed": "প্রদত্ত URL থেকে কন্টেন্ট স্ক্র্যাপ করতে ব্যর্থ। দয়া করে আবার চেষ্টা করুন।",
        "no_content": "বিশ্লেষণের জন্য কোন কন্টেন্ট নেই। দয়া করে একটি ভিন্ন URL চেষ্টা করুন।",
        "analysis_error": "বিশ্লেষণের সময় একটি ত্রুটিoccurred। দয়া করে আবার চেষ্টা করুন।",
        "download_results": "ফলাফল ডাউনলোড করুন",
        "download_html": "HTML হিসাবে ডাউনলোড করুন",
        "enter_first_url": "প্রথম URL লিখুন:",
        "enter_second_url": "দ্বিতীয় URL লিখুন:",
        "comparison_question": "তুলনা প্রশ্ন লিখুন (ঐচ্ছিক):",
        "question_placeholder": "যেমন, 'পণ্য offerings তুলনা করুন' বা 'সমস্ত পণ্য তালিকা করুন'",
        "start_comparison": "তুলনা শুরু করুন",
        "scraping_first": "{} থেকে কন্টেন্ট স্ক্র্যাপ করতে ব্যর্থ। দয়া করে আবার চেষ্টা করুন।",
        "scraping_second": "{} থেকে কন্টেন্ট স্ক্র্যাপ করতে ব্যর্থ। দয়া করে আবার চেষ্টা করুন。",
        "comparison_error": "তুলনার সময় ত্রুটি: {}",
        "enter_both_urls": "দয়া করে তুলনা করার জন্য উভয় URL লিখুন।",
        "user_dashboard": "ব্যবহারকারী ড্যাশবোর্ড",
        "welcome_user": "আপনাকে স্বাগতম, {}!",
        "logout": "লगআউট",
        "language_select": "ভাষা নির্বাচন করুন",
        "content_analysis": "কন্টেন্ট বিশ্লেষণ",
        "summary": "সারাংশ",
        "products": "পণ্য",
        "price_distribution": "মূল্য বন্টন",
        "categories": "বিভাগ",
        "category_distribution": "বিভাগ বন্টন",
        "brands": "ব্র্যান্ড",
        "brand_distribution": "ব্র্যান্ড বন্টন",
        "materials": "উপকরণ",
        "material_distribution": "উপকরণ বন্টন",
        "colors": "রং",
        "color_distribution": "রং বন্টন",
        "sizes": "আকার",
        "size_distribution": "আকার বন্টন",
        "statistics": "পরিসংখ্যান",
        "content_metrics": "কন্টেন্ট মেট্রিক্স",
        "text_length": "টেক্সট দৈর্ঘ্য",
        "total_characters": "মোট অক্ষর",
        "website_comparison": "ওয়েবসাইট তুলনা",
        "comparison_analysis": "তুলনা বিশ্লেষণ",
        "statistics_comparison": "পরিসংখ্যান তুলনা",
        "first_website": "প্রথম ওয়েবسাইট",
        "second_website": "দ্বিতীয় ওয়েবসাইট",
        "metric": "মেট্রिक",
        "products_comparison": "পণ্য তুলনা",
        "price_comparison": "মূল্য বন্টন তুলনা",
        "no_products": "কোন পণ্য পাওয়া যায়নি",
        "pages": "পৃষ্ঠা",
        "images": "ছবি",
        "tables": "টেবিল",
        "lists": "তালিকা",
        "metrics": "মেট্রিক্স",
        "sentiment_analysis": "সেন্টিমেন্ট অ্যানালিসিস",
        "reviews": "রিভিউ",
        "sentiment_distribution": "সেন্টিমেন্ট ডিস্ট্রিবিউশন",
        "positive": "ইতিবাচক",
        "negative": "নেতিবাচক",
        "neutral": "নিরপেক্ষ",
        "average_rating": "গড় রেটিং",
        "review_insights": "রিভিউ ইনসাইটস",
        "similar_projects": "অনুরূপ প্রকল্প",
        "recommended_websites": "প্রস্তাবিত ওয়েবسাইট",
        "similarity_score": "সাদৃশ্য স্কোর",
        "content_similarity": "কন্টেন্ট সাদৃশ্য",
        "product_recommendations": "পণ্য সুপারিশ",
        "view_similar": "অনুরূপ পণ্য দেখুন",
        "based_on_analysis": "আপনার বিশ্লেষণের ইতিহাসের উপর ভিত্তি করে",
        "trending_products": "ট্রেন্ডিং পণ্য",
        "top_rated": "শীর্ষ রেটেড",
        "most_reviewed": "সবচেয়ে পর্যালোচনা",
        "project_suggestions": "প্রকল্প পরামর্শ",
        "review_sentiment": "রিভিউ সেন্টিমেন্ট",
        "product_alternatives": "পণ্য বিকল্প",
        "if_you_liked": "আপনি যদি পছন্দ করেন",
        "here_are_alternatives": "এখানে কিছু বিকল্প আছে",
        "sentiment_score": "সেন্টিমেন্ট স্কোর",
        "overall_sentiment": "সামগ্রিক সেন্টিমেন্ট",
        "review_analysis": "রিভিউ বিশ্লেষণ",
        "recommendation_engine": "সুপারিশ ইঞ্জিন",
        "similar_products": "অনুরূপ পণ্য",
        "better_alternatives": "ভাল বিকল্প",
        "price_range": "মূল্য পরিসীমা",
        "rating_comparison": "রেটিং তুলনা",
        "feature_comparison": "বৈশিষ্ট্য তুলনা",
    }
}

def t(key):
    """Translation helper function"""
    return translations[st.session_state.language].get(key, key)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def initialize_database(json_file_path="users.json"):
    if not os.path.exists(json_file_path):
        with open(json_file_path, "w") as f:
            json.dump({"users": []}, f)

def initialize_user_history():
    if "user_history" not in st.session_state:
        st.session_state.user_history = {
            "analyses": [],
            "comparisons": [],
            "preferences": {}
        }

# Enhanced Sentiment Analysis Class
class SentimentAnalyzer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    def analyze_review_sentiment(self, review_text):
        """Analyze sentiment of a review using Gemini"""
        try:
            prompt = f"""
            Analyze the sentiment of this product review and provide a JSON response with:
            - sentiment: positive, negative, or neutral
            - rating: a number between 1-5 (estimate based on sentiment)
            - confidence: confidence level between 0-1
            - key_phrases: list of key phrases that influenced the sentiment
            - aspects: list of product aspects mentioned (e.g., price, quality, delivery)
            
            Review: {review_text}
            
            Respond with JSON only:
            """
            
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Clean response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Fallback to TextBlob
            return self._fallback_sentiment_analysis(review_text)
    
    def _fallback_sentiment_analysis(self, text):
        """Fallback sentiment analysis using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "positive"
            rating = min(5, max(1, round(3 + polarity * 2)))
        elif polarity < -0.1:
            sentiment = "negative"
            rating = min(5, max(1, round(3 + polarity * 2)))
        else:
            sentiment = "neutral"
            rating = 3
        
        return {
            "sentiment": sentiment,
            "rating": rating,
            "confidence": abs(polarity),
            "key_phrases": blob.noun_phrases[:3],  # Top 3 noun phrases
            "aspects": ["general"]  # Default aspect
        }
    
    def analyze_reviews(self, reviews):
        """Analyze multiple reviews and provide summary"""
        if not reviews:
            return None
        
        sentiments = []
        ratings = []
        all_aspects = []
        
        for review in reviews:
            analysis = self.analyze_review_sentiment(review)
            sentiments.append(analysis['sentiment'])
            ratings.append(analysis['rating'])
            all_aspects.extend(analysis.get('aspects', []))
        
        # Calculate overall metrics
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Count aspect mentions
        aspect_counts = {}
        for aspect in all_aspects:
            aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
        
        return {
            'sentiment_counts': sentiment_counts,
            'average_rating': avg_rating,
            'total_reviews': len(reviews),
            'aspect_counts': aspect_counts,
            'individual_analyses': list(zip(reviews, sentiments, ratings))
        }

# Enhanced Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    def calculate_product_similarity(self, product1, product2):
        """Calculate similarity between two products"""
        # Create feature vectors for comparison
        features1 = f"{product1.get('name', '')} {product1.get('description', '')} {product1.get('category', '')} {product1.get('brand', '')}"
        features2 = f"{product2.get('name', '')} {product2.get('description', '')} {product2.get('category', '')} {product2.get('brand', '')}"
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform([features1, features2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0
    
    def find_similar_products(self, target_product, all_products, top_n=5):
        """Find similar products based on features"""
        similarities = []
        
        for product in all_products:
            if product != target_product:
                similarity = self.calculate_product_similarity(target_product, product)
                similarities.append({
                    'product': product,
                    'similarity': similarity
                })
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_n]
    
    def find_better_alternatives(self, target_product, all_products, top_n=3):
        """Find better alternatives (higher rating, similar price)"""
        alternatives = []
        
        for product in all_products:
            if product != target_product:
                # Check if product has higher rating
                target_rating = float(target_product.get('rating', 0)) if target_product.get('rating') else 0
                product_rating = float(product.get('rating', 0)) if product.get('rating') else 0
                
                # Check if price is similar (±20%)
                target_price = self._parse_price(target_product.get('price', '0'))
                product_price = self._parse_price(product.get('price', '0'))
                
                price_similar = False
                if target_price > 0 and product_price > 0:
                    price_ratio = product_price / target_price
                    price_similar = 0.8 <= price_ratio <= 1.2
                
                if product_rating > target_rating and price_similar:
                    alternatives.append({
                        'product': product,
                        'rating_improvement': product_rating - target_rating,
                        'price_difference': product_price - target_price
                    })
        
        # Sort by rating improvement
        alternatives.sort(key=lambda x: x['rating_improvement'], reverse=True)
        return alternatives[:top_n]
    
    def _parse_price(self, price_str):
        """Parse price string to float"""
        if not price_str:
            return 0
        
        try:
            # Remove currency symbols and commas
            price_str = re.sub(r'[^\d.]', '', str(price_str))
            return float(price_str)
        except:
            return 0

# Enhanced Project Similarity Engine
class SimilarityEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.analysis_history = []
        self.recommendation_engine = RecommendationEngine()
    
    def add_analysis(self, analysis, url):
        """Add analysis to history for future comparisons"""
        self.analysis_history.append({
            'url': url,
            'analysis': analysis,
            'timestamp': time.time()
        })
    
    def calculate_similarity(self, analysis1, analysis2):
        """Calculate similarity between two analyses"""
        # Combine text data for comparison
        text1 = " ".join([
            analysis1.get('summary', ''),
            " ".join(analysis1.get('key_topics', [])),
            " ".join(analysis1.get('categories', [])),
            " ".join([p.get('name', '') for p in analysis1.get('products', [])]),
            " ".join([p.get('description', '') for p in analysis1.get('products', [])])
        ])
        
        text2 = " ".join([
            analysis2.get('summary', ''),
            " ".join(analysis2.get('key_topics', [])),
            " ".join(analysis2.get('categories', [])),
            " ".join([p.get('name', '') for p in analysis2.get('products', [])]),
            " ".join([p.get('description', '') for p in analysis2.get('products', [])])
        ])
        
        # Vectorize and calculate similarity
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0
    
    def find_similar_websites(self, current_analysis, current_url, top_n=3):
        """Find similar websites from history"""
        similarities = []
        
        for history_item in self.analysis_history:
            if history_item['url'] != current_url:  # Don't compare with self
                similarity = self.calculate_similarity(current_analysis, history_item['analysis'])
                similarities.append({
                    'url': history_item['url'],
                    'similarity': similarity,
                    'timestamp': history_item['timestamp']
                })
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_n]
    
    def get_all_products(self):
        """Get all products from analysis history"""
        all_products = []
        for analysis in self.analysis_history:
            all_products.extend(analysis['analysis'].get('products', []))
        return all_products
    
    def get_product_recommendations(self, target_product):
        """Get product recommendations for a target product"""
        all_products = self.get_all_products()
        
        similar_products = self.recommendation_engine.find_similar_products(target_product, all_products)
        better_alternatives = self.recommendation_engine.find_better_alternatives(target_product, all_products)
        
        return {
            'similar_products': similar_products,
            'better_alternatives': better_alternatives
        }

# Enhanced WebScraper with review extraction
class WebScraper:
    def __init__(self, max_depth: int = 2, max_pages: int = 50, max_workers: int = 5):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.visited_urls = set()
        self.domain_blacklist = {
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'youtube.com', 'pinterest.com', 'tiktok.com', 'reddit.com'
        }
        self.content_types = {
            'text': ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div', 'article', 'section'],
            'links': ['a'],
            'images': ['img'],
            'tables': ['table'],
            'lists': ['ul', 'ol', 'li'],
            'reviews': ['div', 'span']  # For review content
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }

    def extract_reviews(self, soup):
        """Extract product reviews from page"""
        reviews = []
        
        # Look for common review containers
        review_selectors = [
            {'class': 'review'},
            {'class': 'customer-review'},
            {'class': 'user-review'},
            {'class': 'testimonial'},
            {'itemprop': 'review'},
            {'data-hook': 'review'},
        ]
        
        for selector in review_selectors:
            review_elements = soup.find_all('div', selector)
            for review in review_elements:
                review_text = review.get_text(' ', strip=True)
                if review_text and len(review_text) > 20:  # Minimum review length
                    reviews.append(review_text)
        
        return reviews

    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        try:
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_domain)
            
            # Skip if URL is in blacklist
            if any(domain in parsed_url.netloc for domain in self.domain_blacklist):
                return False
                
            # Allow same domain and subdomains
            return (parsed_url.netloc == parsed_base.netloc or 
                   parsed_url.netloc.endswith('.' + parsed_base.netloc))
        except:
            return False

    def extract_content(self, url: str, max_retries: int = 3) -> Dict:
        """Extract structured content from a URL with retry logic."""
        for attempt in range(max_retries):
            try:
                with requests.Session() as session:
                    session.headers.update(self.headers)
                    response = session.get(url, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    content = {
                        'text': '',
                        'links': set(),
                        'images': set(),
                        'tables': [],
                        'lists': [],
                        'products': [],
                        'reviews': self.extract_reviews(soup)  # Extract reviews
                    }
                    
                    # Extract text content
                    for tag in self.content_types['text']:
                        for element in soup.find_all(tag):
                            content['text'] += element.get_text(strip=True) + ' '
                    
                    # Enhanced product extraction
                    # Look for products in data attributes and specific product containers
                    product_elements = soup.find_all(['div', 'article', 'section'], 
                        class_=lambda x: x and any(cls in str(x).lower() for cls in ['product', 'item', 'card', 'grid', 'product-tile']))
                    
                    for product_elem in product_elements:
                        product_data = {}
                        
                        # Extract from data attributes first
                        if product_elem.has_attr('data-tileanalyticdata'):
                            try:
                                data = json.loads(product_elem['data-tileanalyticdata'])
                                product_data.update({
                                    'name': data.get('name', ''),
                                    'id': data.get('id', ''),
                                    'price': data.get('price', ''),
                                    'brand': data.get('brand', ''),
                                    'category': data.get('category', ''),
                                    'variant': data.get('variant', '')
                                })
                            except json.JSONDecodeError:
                                pass
                        
                        # Extract from data-tileanalyticdatag4 if available
                        if product_elem.has_attr('data-tileanalyticdatag4'):
                            try:
                                data = json.loads(product_elem['data-tileanalyticdatag4'])
                                product_data.update({
                                    'name': product_data.get('name', '') or data.get('item_name', ''),
                                    'id': product_data.get('id', '') or data.get('item_id', ''),
                                    'price': product_data.get('price', '') or data.get('price', ''),
                                    'brand': product_data.get('brand', '') or data.get('item_brand', ''),
                                    'category': product_data.get('category', '') or data.get('item_category', ''),
                                    'variant': product_data.get('variant', '') or data.get('item_variant', ''),
                                    'category2': data.get('item_category2', ''),
                                    'category3': data.get('item_category3', '')
                                })
                            except json.JSONDecodeError:
                                pass
                        
                        # Extract from HTML elements
                        # Product name and category from pdp-link
                        pdp_link = product_elem.find('div', class_='pdp-link')
                        if pdp_link:
                            category_elem = pdp_link.find('p', class_='mb-0')
                            if category_elem:
                                product_data['category'] = product_data.get('category', '') or category_elem.get_text(strip=True)
                            
                            name_elem = pdp_link.find('a', class_='link')
                            if name_elem:
                                product_data['name'] = product_data.get('name', '') or name_elem.get_text(strip=True)
                        
                        # Extract price from product-price-promotion
                        price_container = product_elem.find('div', class_='product-price-promotion')
                        if price_container:
                            # Look for price in sales span
                            price_elem = price_container.find('span', class_='sales')
                            if price_elem:
                                # Try to get price from content attribute
                                price = price_elem.find('span', attrs={'content': True})
                                if price:
                                    product_data['price'] = price['content']
                                else:
                                    # Try to get price from text
                                    price_text = price_elem.get_text(strip=True)
                                    price_match = re.search(r'₹?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', price_text)
                                    if price_match:
                                        product_data['price'] = price_match.group(1)
                        
                        # Extract images from main-image-slider
                        images = []
                        image_slider = product_elem.find('div', class_='main-image-slider')
                        if image_slider:
                            for img in image_slider.find_all('img'):
                                src = img.get('src') or img.get('data-src')
                                if src:
                                    full_url = urljoin(url, src)
                                    images.append(full_url)
                                    content['images'].add(full_url)
                        
                        # Extract color swatches
                        colors = []
                        swatches = product_elem.find('div', class_='color-swatches')
                        if swatches:
                            for swatch in swatches.find_all('img', class_='swatch'):
                                color_name = swatch.get('alt', '').split(',')[-1].strip()
                                if color_name:
                                    colors.append(color_name)
                        
                        # Extract product status (New, Sale, etc.)
                        status_tags = product_elem.find_all('span', class_=lambda x: x and any(cls in str(x).lower() for cls in ['new-tag', 'sale-tag']))
                        if status_tags:
                            product_data['status'] = [tag.get_text(strip=True) for tag in status_tags]
                        
                        # Extract quick view data
                        quickview = product_elem.find('a', class_='quickview')
                        if quickview and quickview.has_attr('href'):
                            product_data['quickview_url'] = urljoin(url, quickview['href'])
                        
                        # Extract wishlist data
                        wishlist = product_elem.find('a', class_='wishlistTile')
                        if wishlist and wishlist.has_attr('href'):
                            product_data['wishlist_url'] = urljoin(url, wishlist['href'])
                        
                        # Add extracted data to product
                        if images:
                            product_data['images'] = images
                        if colors:
                            product_data['colors'] = colors
                        
                        # Add product to list if we found any data
                        if product_data:
                            content['products'].append(product_data)
                    
                    # Extract links
                    for link in soup.find_all('a'):
                        href = link.get('href')
                        if href:
                            full_url = urljoin(url, href)
                            content['links'].add(full_url)
                    
                    # Extract images (non-product)
                    for img in soup.find_all('img'):
                        src = img.get('src') or img.get('data-src')
                        if src:
                            full_url = urljoin(url, src)
                            content['images'].add(full_url)
                    
                    # Extract tables
                    for table in soup.find_all('table'):
                        content['tables'].append(str(table))
                    
                    # Extract lists
                    for list_type in self.content_types['lists']:
                        for list_elem in soup.find_all(list_type):
                            content['lists'].append(str(list_elem))
                    
                    return content
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All attempts failed for {url}: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error extracting content from {url}: {e}")
                return None

    def scrape_website(self, start_url: str) -> Dict:
        """Scrape website content with improved logic."""
        base_domain = urlparse(start_url).netloc
        queue = deque([(start_url, 0)])
        all_content = {
            'text': '',
            'links': set(),
            'images': set(),
            'tables': [],
            'lists': [],
            'products': [],
            'reviews': []
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while queue and len(self.visited_urls) < self.max_pages:
                current_url, depth = queue.popleft()
                
                if current_url in self.visited_urls or depth > self.max_depth:
                    continue
                    
                self.visited_urls.add(current_url)
                
                # Submit URL for processing
                future = executor.submit(self.extract_content, current_url)
                content = future.result()
                
                if content:
                    # Merge content
                    all_content['text'] += content['text'] + ' '
                    all_content['links'].update(content['links'])
                    all_content['images'].update(content['images'])
                    all_content['tables'].extend(content['tables'])
                    all_content['lists'].extend(content['lists'])
                    all_content['products'].extend(content['products'])
                    all_content['reviews'].extend(content['reviews'])
                    
                    # Add new links to queue
                    for link in content['links']:
                        if (self.is_valid_url(link, base_domain) and 
                            link not in self.visited_urls and 
                            len(self.visited_urls) < self.max_pages):
                            queue.append((link, depth + 1))
        
        return all_content

# Enhanced ContentAnalyzer with sentiment analysis and recommendation engine
class ContentAnalyzer:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.similarity_engine = SimilarityEngine()
        
    def process_text(self, text: str) -> List[str]:
        """Process and chunk text for analysis."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=800
        )
        return text_splitter.split_text(text)
    
    def create_vector_store(self, chunks: List[str]):
        """Create vector store from text chunks."""
        vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    
    def analyze_content(self, content: Dict, query: str = None, url: str = None) -> Dict:
        """Analyze scraped content using Gemini with enhanced detail extraction."""
        analysis = {
            'summary': '',
            'key_topics': [],
            'sentiment': '',
            'entities': [],
            'statistics': {},
            'products': content.get('products', []),
            'categories': [],
            'prices': [],
            'features': [],
            'specifications': [],
            'reviews': content.get('reviews', []),
            'ratings': [],
            'availability': [],
            'brands': [],
            'materials': [],
            'colors': [],
            'sizes': [],
            'review_analysis': None,
            'url': url
        }
        
        # Analyze reviews if any
        if analysis['reviews']:
            analysis['review_analysis'] = self.sentiment_analyzer.analyze_reviews(analysis['reviews'])
        
        # Generate detailed analysis
        prompt = f"""
        Analyze the following content{f" focusing on: {query}" if query else ""} and provide a comprehensive analysis.
        Extract as much detail as possible about:
        1. Products (name, description, price, features, specifications)
        2. Categories and subcategories
        3. Product features and specifications
        4. Customer reviews and ratings
        5. Product availability
        6. Brands mentioned
        7. Materials used
        8. Available colors
        9. Available sizes
        10. Key topics and themes
        11. Overall sentiment
        12. Important entities
        
        Format your response as a JSON object with this exact structure:
        {{
            "summary": "detailed summary here",
            "products": [
                {{
                    "name": "product name",
                    "description": "detailed description",
                    "price": "price if available",
                    "features": ["feature1", "feature2"],
                    "specifications": ["spec1", "spec2"],
                    "rating": "rating if available",
                    "availability": "in stock/out of stock",
                    "brand": "brand name",
                    "material": "material type",
                    "color": "color",
                    "size": "size"
                }}
            ],
            "categories": ["main category", "subcategory1", "subcategory2"],
            "topics": ["topic1", "topic2"],
            "sentiment": "positive/negative/neutral",
            "entities": ["entity1", "entity2"],
            "brands": ["brand1", "brand2"],
            "materials": ["material1", "material2"],
            "colors": ["color1", "color2"],
            "sizes": ["size1", "size2"]
        }}
        
        Content: {content['text'][:10000]}  # Limit text length for analysis
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Clean the response to ensure valid JSON
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse the JSON response
            detailed_analysis = json.loads(response_text)
            
            # Update analysis with the parsed data
            analysis.update({
                'summary': detailed_analysis.get('summary', ''),
                'products': analysis['products'] + detailed_analysis.get('products', []),  # Combine scraped and analyzed products
                'categories': detailed_analysis.get('categories', []),
                'key_topics': detailed_analysis.get('topics', []),
                'sentiment': detailed_analysis.get('sentiment', ''),
                'entities': detailed_analysis.get('entities', []),
                'brands': detailed_analysis.get('brands', []),
                'materials': detailed_analysis.get('materials', []),
                'colors': detailed_analysis.get('colors', []),
                'sizes': detailed_analysis.get('sizes', [])
            })
            
            # Calculate basic statistics
            analysis['statistics'] = {
                'total_pages': len(content['links']),
                'total_images': len(content['images']),
                'total_tables': len(content['tables']),
                'total_lists': len(content['lists']),
                'text_length': len(content['text']),
                'total_products': len(analysis['products']),  # Use combined products count
                'total_categories': len(detailed_analysis.get('categories', [])),
                'total_brands': len(detailed_analysis.get('brands', [])),
                'total_materials': len(detailed_analysis.get('materials', [])),
                'total_colors': len(detailed_analysis.get('colors', [])),
                'total_sizes': len(detailed_analysis.get('sizes', [])),
                'total_reviews': len(analysis['reviews'])
            }
            
            expected_stats = [
                'total_pages', 'total_images', 'total_tables', 'total_lists', 'text_length',
                'total_products', 'total_categories', 'total_brands', 'total_materials', 'total_colors', 'total_sizes', 'total_reviews'
            ]
            for stat in expected_stats:
                if stat not in analysis['statistics']:
                    analysis['statistics'][stat] = 0
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {response.content if 'response' in locals() else 'No response'}")
            analysis['summary'] = "Error parsing analysis results. Please try again."
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {e}")
            analysis['summary'] = "Error in content analysis. Please try again."
        
        # Add to similarity engine for future comparisons
        if url:
            self.similarity_engine.add_analysis(analysis, url)
        
        return analysis

    def get_similar_projects(self, current_analysis, current_url):
        """Get similar projects based on current analysis"""
        return self.similarity_engine.find_similar_websites(current_analysis, current_url)
    
    def get_product_recommendations(self, target_product):
        """Get product recommendations for a target product"""
        return self.similarity_engine.get_product_recommendations(target_product)

    def compare_websites(self, analysis1: Dict, analysis2: Dict, question: str = None):
        """Compare two websites based on analysis results."""
        st.subheader(t("website_comparison"))
        
        if question:
            # Generate comparison based on specific question
            prompt = f"""
            Compare the following two websites based on this question: {question}
            
            Website 1:
            {json.dumps(analysis1, indent=2)}
            
            Website 2:
            {json.dumps(analysis2, indent=2)}
            
            Provide a detailed comparison focusing on the question asked.
            """
            
            try:
                response = self.llm.invoke(prompt)
                st.write("### " + t("comparison_analysis"))
                st.write(response.content)
            except Exception as e:
                st.error(f"Error generating comparison: {e}")
        
        # Statistics comparison with improved visualization
        st.write("### " + t("statistics_comparison"))
        comparison_data = {
            t('metric'): [t('pages'), t('images'), t('tables'), t('lists'), t('products'), t('categories'), 
                      t('brands'), t('materials'), t('colors'), t('sizes'), t('reviews')],
            t('first_website'): [
                analysis1['statistics'].get('total_pages', 0),
                analysis1['statistics'].get('total_images', 0),
                analysis1['statistics'].get('total_tables', 0),
                analysis1['statistics'].get('total_lists', 0),
                analysis1['statistics'].get('total_products', 0),
                analysis1['statistics'].get('total_categories', 0),
                analysis1['statistics'].get('total_brands', 0),
                analysis1['statistics'].get('total_materials', 0),
                analysis1['statistics'].get('total_colors', 0),
                analysis1['statistics'].get('total_sizes', 0),
                analysis1['statistics'].get('total_reviews', 0)
            ],
            t('second_website'): [
                analysis2['statistics'].get('total_pages', 0),
                analysis2['statistics'].get('total_images', 0),
                analysis2['statistics'].get('total_tables', 0),
                analysis2['statistics'].get('total_lists', 0),
                analysis2['statistics'].get('total_products', 0),
                analysis2['statistics'].get('total_categories', 0),
                analysis2['statistics'].get('total_brands', 0),
                analysis2['statistics'].get('total_materials', 0),
                analysis2['statistics'].get('total_colors', 0),
                analysis2['statistics'].get('total_sizes', 0),
                analysis2['statistics'].get('total_reviews', 0)
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create a grid of comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### " + t("content_metrics") + " " + t("comparison"))
            fig, ax = plt.subplots(figsize=(10, 6))
            comparison_df.plot(kind='bar', x=t('metric'), ax=ax)
            ax.set_xlabel(t('metrics'))
            ax.set_ylabel('Count')
            ax.set_title(t('content_metrics') + ' ' + t("comparison"))
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.write("#### " + t("text_length") + " " + t("comparison"))
            fig, ax = plt.subplots(figsize=(8, 6))
            text_lengths = pd.DataFrame({
                'Website': [t('first_website'), t('second_website')],
                t('text_length'): [
                    analysis1['statistics'].get('text_length', 0),
                    analysis2['statistics'].get('text_length', 0)
                ]
            })
            text_lengths.plot(kind='bar', x='Website', y=t('text_length'), ax=ax)
            ax.set_ylabel('Characters')
            ax.set_title(t('text_length') + ' ' + t("comparison"))
            st.pyplot(fig)
        
        # Product comparison if available
        if analysis1['products'] or analysis2['products']:
            st.write("### " + t("products_comparison"))
            
            # Create product DataFrames
            products1_df = pd.DataFrame(analysis1['products'])
            products2_df = pd.DataFrame(analysis2['products'])
            
            if not products1_df.empty or not products2_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### " + t("first_website") + " " + t("products"))
                    if not products1_df.empty:
                        st.dataframe(products1_df)
                    else:
                        st.write(t("no_products"))
                
                with col2:
                    st.write("#### " + t("second_website") + " " + t("products"))
                    if not products2_df.empty:
                        st.dataframe(products2_df)
                    else:
                        st.write(t("no_products"))
                
                # Price comparison if prices are available
                if 'price' in products1_df.columns and 'price' in products2_df.columns:
                    try:
                        # Clean price columns
                        products1_df['price'] = products1_df['price'].str.replace('₹', '').str.replace(',', '').str.strip()
                        products2_df['price'] = products2_df['price'].str.replace('₹', '').str.replace(',', '').str.strip()
                        products1_df['price'] = pd.to_numeric(products1_df['price'], errors='coerce')
                        products2_df['price'] = pd.to_numeric(products2_df['price'], errors='coerce')
                        
                        if not products1_df['price'].isna().all() and not products2_df['price'].isna().all():
                            st.write("#### " + t("price_comparison"))
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            products1_df['price'].hist(ax=ax1, bins=20)
                            ax1.set_xlabel('Price (₹)')
                            ax1.set_ylabel('Number of Products')
                            ax1.set_title(t('first_website') + ' ' + t("price_distribution"))
                            
                            products2_df['price'].hist(ax=ax2, bins=20)
                            ax2.set_xlabel('Price (₹)')
                            ax2.set_ylabel('Number of Products')
                            ax2.set_title(t('second_website') + ' ' + t("price_distribution"))
                            
                            st.pyplot(fig)
                    except Exception as e:
                        logger.error(f"Error creating price comparison: {e}")
        
        # Review comparison if available
        if analysis1.get('review_analysis') or analysis2.get('review_analysis'):
            st.write("### " + t("sentiment_analysis") + " " + t("comparison"))
            
            col1, col2 = st.columns(2)
            
            with col1:
                if analysis1.get('review_analysis'):
                    st.write("#### " + t("first_website") + " " + t("sentiment_analysis"))
                    sentiment_counts = analysis1['review_analysis']['sentiment_counts']
                    fig = px.pie(
                        values=list(sentiment_counts.values()),
                        names=list(sentiment_counts.keys()),
                        title=t("sentiment_distribution")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric(t("average_rating"), f"{analysis1['review_analysis']['average_rating']:.1f}/5")
                    st.metric(t("reviews"), analysis1['review_analysis']['total_reviews'])
            
            with col2:
                if analysis2.get('review_analysis'):
                    st.write("#### " + t("second_website") + " " + t("sentiment_analysis"))
                    sentiment_counts = analysis2['review_analysis']['sentiment_counts']
                    fig = px.pie(
                        values=list(sentiment_counts.values()),
                        names=list(sentiment_counts.keys()),
                        title=t("sentiment_distribution")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric(t("average_rating"), f"{analysis2['review_analysis']['average_rating']:.1f}/5")
                    st.metric(t("reviews"), analysis2['review_analysis']['total_reviews'])

        comparison_summary = ""

        # Add question-based comparison (if any)
        if question:
            try:
                response = self.llm.invoke(prompt)
                question_response = response.content.replace('\n', '<br>')
                comparison_summary += f"<h3>Comparison Based on Question</h3><p>{question_response}</p>"
            except Exception as e:
                comparison_summary += f"<h3>Comparison Based on Question</h3><p>Error generating comparison: {e}</p>"
        else:
            comparison_summary += "<h3>Comparison Based on Question</h3><p>No specific question was asked.</p>"

        # Add statistics comparison summary
        comparison_summary += "<h3>Statistics Comparison</h3>"
        comparison_summary += """
        <table border="1" cellpadding="6" cellspacing="0">
        <tr><th>Metric</th><th>First Website</th><th>Second Website</th></tr>
        """
        for i in range(len(comparison_data[t('metric')])):
            metric = comparison_data[t('metric')][i]
            first = comparison_data[t('first_website')][i]
            second = comparison_data[t('second_website')][i]
            comparison_summary += f"<tr><td>{metric}</td><td>{first}</td><td>{second}</td></tr>"
        comparison_summary += "</table>"

        # Add text length comparison
        text1 = analysis1['statistics'].get('text_length', 0)
        text2 = analysis2['statistics'].get('text_length', 0)
        comparison_summary += """
        <h3>Text Length Comparison</h3>
        <ul>
        <li>First Website Text Length: {} characters</li>
        <li>Second Website Text Length: {} characters</li>
        </ul>
        """.format(text1, text2)

        # Return summary
        return comparison_summary

# Enhanced visualization with sentiment analysis and recommendations
def visualize_analysis(analysis: Dict, query: str = None):
    """Create detailed visualizations from analysis results."""
    st.subheader(t("content_analysis"))
    
    # Summary
    st.write("### " + t("summary"))
    st.write(analysis['summary'])
    
    # Sentiment Analysis Section
    if analysis.get('review_analysis'):
        st.write("### " + t("sentiment_analysis"))
        review_analysis = analysis['review_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = review_analysis['sentiment_counts']
            fig = px.pie(
                values=list(sentiment_counts.values()),
                names=list(sentiment_counts.keys()),
                title=t("sentiment_distribution")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average rating
            st.metric(t("average_rating"), f"{review_analysis['average_rating']:.1f}/5")
            st.metric(t("reviews"), review_analysis['total_reviews'])
            
            # Sentiment insights
            st.write("#### " + t("review_insights"))
            if sentiment_counts['positive'] > sentiment_counts['negative']:
                st.success("Overall positive sentiment from customers")
            else:
                st.warning("Mixed or negative sentiment from customers")
        
        # Show aspect analysis if available
        if review_analysis.get('aspect_counts'):
            st.write("#### " + t("review_analysis"))
            aspects_df = pd.DataFrame({
                'Aspect': list(review_analysis['aspect_counts'].keys()),
                'Mentions': list(review_analysis['aspect_counts'].values())
            })
            aspects_df = aspects_df.sort_values('Mentions', ascending=False)
            
            fig = px.bar(aspects_df, x='Aspect', y='Mentions', 
                        title="Most Discussed Product Aspects")
            st.plotly_chart(fig, use_container_width=True)
        
        # Show sample reviews with sentiment
        st.write("#### Sample Reviews")
        sample_reviews = review_analysis['individual_analyses'][:5]  # Show first 5
        for review, sentiment, rating in sample_reviews:
            sentiment_color = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'gray'
            }.get(sentiment, 'gray')
            
            st.markdown(f"""
            <div style="border-left: 4px solid {sentiment_color}; padding-left: 10px; margin: 10px 0;">
                <p>{review}</p>
                <p><strong>Sentiment:</strong> <span style="color: {sentiment_color}">{sentiment}</span> | 
                <strong>Rating:</strong> {rating}/5</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Products
    if analysis['products']:
        st.write("### " + t("products"))
        # Convert lists to strings in product data
        processed_products = []
        for product in analysis['products']:
            processed_product = {}
            for key, value in product.items():
                if isinstance(value, list):
                    processed_product[key] = ', '.join(str(v) for v in value)
                else:
                    processed_product[key] = str(value) if value is not None else ''
            processed_products.append(processed_product)
        
        products_df = pd.DataFrame(processed_products)
        if not products_df.empty:
            st.dataframe(products_df)
            
            # Product price distribution if prices are available
            if 'price' in products_df.columns:
                try:
                    # Clean price column
                    products_df['price'] = products_df['price'].str.replace('₹', '').str.replace(',', '').str.strip()
                    products_df['price'] = pd.to_numeric(products_df['price'], errors='coerce')
                    
                    if not products_df['price'].isna().all():
                        st.write("### " + t("price_distribution"))
                        fig, ax = plt.subplots(figsize=(10, 6))
                        products_df['price'].hist(ax=ax, bins=20)
                        ax.set_xlabel('Price (₹)')
                        ax.set_ylabel('Number of Products')
                        ax.set_title(t("price_distribution"))
                        st.pyplot(fig)
                except Exception as e:
                    logger.error(f"Error creating price distribution: {e}")
    
    # Categories
    if analysis['categories']:
        st.write("### " + t("categories"))
        categories_df = pd.DataFrame({'Category': analysis['categories']})
        st.dataframe(categories_df)
        
        # Category distribution
        st.write("### " + t("category_distribution"))
        fig, ax = plt.subplots(figsize=(10, 6))
        category_counts = categories_df['Category'].value_counts()
        ax.pie([1] * len(category_counts), labels=category_counts.index, autopct='')
        ax.set_title(t("categories"))
        st.pyplot(fig)
    
    # Brands
    if analysis['brands']:
        st.write("### " + t("brands"))
        brands_df = pd.DataFrame({'Brand': analysis['brands']})
        st.dataframe(brands_df)
        
        # Brand distribution
        st.write("### " + t("brand_distribution"))
        fig, ax = plt.subplots(figsize=(10, 6))
        brand_counts = brands_df['Brand'].value_counts()
        ax.pie([1] * len(brand_counts), labels=brand_counts.index, autopct='')
        ax.set_title(t("brands"))
        st.pyplot(fig)
    
    # Materials
    if analysis['materials']:
        st.write("### " + t("materials"))
        # Convert list of materials to string if it's a list
        materials = analysis['materials']
        if isinstance(materials, list):
            materials = [', '.join(m) if isinstance(m, list) else str(m) for m in materials]
        materials_df = pd.DataFrame({'Material': materials})
        st.dataframe(materials_df)
        
        # Material distribution
        st.write("### " + t("material_distribution"))
        fig, ax = plt.subplots(figsize=(10, 6))
        material_counts = materials_df['Material'].value_counts()
        ax.pie([1] * len(material_counts), labels=material_counts.index, autopct='')
        ax.set_title(t("materials"))
        st.pyplot(fig)
    
    # Colors
    if analysis['colors']:
        st.write("### " + t("colors"))
        # Convert list of colors to string if it's a list
        colors = analysis['colors']
        if isinstance(colors, list):
            colors = [', '.join(c) if isinstance(c, list) else str(c) for c in colors]
        colors_df = pd.DataFrame({'Color': colors})
        st.dataframe(colors_df)
        
        # Color distribution
        st.write("### " + t("color_distribution"))
        fig, ax = plt.subplots(figsize=(10, 6))
        color_counts = colors_df['Color'].value_counts()
        ax.pie([1] * len(color_counts), labels=color_counts.index, autopct='')
        ax.set_title(t("colors"))
        st.pyplot(fig)
    
    # Sizes
    if analysis['sizes']:
        st.write("### " + t("sizes"))
        # Convert list of sizes to string if it's a list
        sizes = analysis['sizes']
        if isinstance(sizes, list):
            sizes = [', '.join(s) if isinstance(s, list) else str(s) for s in sizes]
        sizes_df = pd.DataFrame({'Size': sizes})
        st.dataframe(sizes_df)
        
        # Size distribution
        st.write("### " + t("size_distribution"))
        fig, ax = plt.subplots(figsize=(10, 6))
        size_counts = sizes_df['Size'].value_counts()
        ax.pie([1] * len(size_counts), labels=size_counts.index, autopct='')
        ax.set_title(t("sizes"))
        st.pyplot(fig)
    
    # Statistics with improved visualization
    st.write("### " + t("statistics"))
    stats_data = {
        t['metric']: [t('pages'), t('images'), t('tables'), t('lists'), t('products'), t('categories'), 
                  t('brands'), t('materials'), t('colors'), t('sizes'), t('reviews')],
        'Count': [
            analysis['statistics'].get('total_pages', 0),
            analysis['statistics'].get('total_images', 0),
            analysis['statistics'].get('total_tables', 0),
            analysis['statistics'].get('total_lists', 0),
            analysis['statistics'].get('total_products', 0),
            analysis['statistics'].get('total_categories', 0),
            analysis['statistics'].get('total_brands', 0),
            analysis['statistics'].get('total_materials', 0),
            analysis['statistics'].get('total_colors', 0),
            analysis['statistics'].get('total_sizes', 0),
            analysis['statistics'].get('total_reviews', 0)
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create a grid of charts
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### " + t("content_metrics"))
        fig, ax = plt.subplots(figsize=(10, 6))
        stats_df.plot(kind='bar', x=t('metric'), y='Count', ax=ax)
        ax.set_xlabel(t('metrics'))
        ax.set_ylabel('Count')
        ax.set_title(t("content_metrics"))
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.write("#### " + t("text_length"))
        st.metric(t("total_characters"), f"{analysis['statistics'].get('text_length', 0):,}")

def download_as_html(analysis, filename="analysis_results.html"):
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_base64

    html = [
        "<html><head><title>Web Analysis Results</title><style>body{font-family:Arial,sans-serif;margin:20px;}h1,h2{color:#333;}table{border-collapse:collapse;width:100%;margin-bottom:20px;}th,td{border:1px solid #ddd;padding:8px;}th{background:#f2f2f2;}section{margin-bottom:30px;}img{max-width:100%;height:auto;}</style></head><body>"
    ]
    html.append("<h1>Web Analysis Results</h1>")
    # Summary
    html.append("<section><h2>Summary</h2><p>{}</p></section>".format(analysis.get('summary', '')))
    # Statistics
    stats = analysis.get('statistics', {})
    if stats:
        html.append("<section><h2>Statistics</h2><table><tr>{}</tr><tr>{}</tr></table></section>".format(
            ''.join(f"<th>{k.replace('_',' ').title()}</th>" for k in stats.keys()),
            ''.join(f"<td>{v}</td>" for v in stats.values())
        ))
    # Products
    products = analysis.get('products', [])
    if products:
        html.append("<section><h2>Products</h2><table><tr>{}</tr>".format(
            ''.join(f"<th>{k.title()}</th>" for k in products[0].keys())
        ))
        for prod in products:
            html.append("<tr>{}</tr>".format(''.join(f"<td>{str(v)}</td>" for v in prod.values())))
        html.append("</table></section>")
    # Categories
    categories = analysis.get('categories', [])
    if categories:
        html.append("<section><h2>Categories</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{c}</li>" for c in categories)
        ))
    # Brands
    brands = analysis.get('brands', [])
    if brands:
        html.append("<section><h2>Brands</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{b}</li>" for b in brands)
        ))
    # Materials
    materials = analysis.get('materials', [])
    if materials:
        html.append("<section><h2>Materials</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{m}</li>" for m in materials)
        ))
    # Colors
    colors = analysis.get('colors', [])
    if colors:
        html.append("<section><h2>Colors</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{c}</li>" for c in colors)
        ))
    # Sizes
    sizes = analysis.get('sizes', [])
    if sizes:
        html.append("<section><h2>Sizes</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{s}</li>" for s in sizes)
        ))
    # Reviews and Sentiment Analysis
    if analysis.get('review_analysis'):
        review_analysis = analysis['review_analysis']
        html.append("<section><h2>Sentiment Analysis</h2>")
        html.append(f"<p>Average Rating: {review_analysis['average_rating']:.1f}/5</p>")
        html.append(f"<p>Total Reviews: {review_analysis['total_reviews']}</p>")
        html.append("<h3>Sentiment Distribution</h3><ul>")
        for sentiment, count in review_analysis['sentiment_counts'].items():
            html.append(f"<li>{sentiment.title()}: {count}</li>")
        html.append("</ul></section>")
    # Price Distribution Graph (if available)
    if products and any('price' in p for p in products):
        import pandas as pd
        prices = [float(p['price']) for p in products if p.get('price') and str(p['price']).replace('.','',1).isdigit()]
        if prices:
            fig, ax = plt.subplots(figsize=(8,4))
            pd.Series(prices).hist(ax=ax, bins=20)
            ax.set_xlabel('Price')
            ax.set_ylabel('Count')
            ax.set_title('Product Price Distribution')
            img_base64 = fig_to_base64(fig)
            html.append(f'<section><h2>Price Distribution</h2><img src="data:image/png;base64,{img_base64}"/></section>')
    # Add more graphs as needed (e.g., category/brand pie charts)
    # End
    html.append("</body></html>")
    return ''.join(html)

# New function to show similar projects
def show_similar_projects(analysis, url):
    """Display similar projects based on current analysis"""
    st.write("### " + t("similar_projects"))
    
    similar_projects = content_analyzer.get_similar_projects(analysis, url)
    
    if similar_projects:
        st.write(t("based_on_analysis"))
        
        for project in similar_projects:
            similarity_percent = project['similarity'] * 100
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0;">
                <h4>{project['url']}</h4>
                <p><strong>{t('similarity_score')}:</strong> {similarity_percent:.1f}%</p>
                <p><strong>{t('content_similarity')}:</strong> {"High" if similarity_percent > 70 else "Medium" if similarity_percent > 40 else "Low"}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No similar projects found in history. Analyze more websites to get recommendations.")

# Enhanced function to show product recommendations
def show_product_recommendations(analysis):
    """Display product recommendations based on current analysis"""
    if analysis.get('products'):
        st.write("### " + t("product_recommendations"))
        
        # Get top rated products
        products_with_ratings = []
        for product in analysis['products']:
            if 'rating' in product and isinstance(product['rating'], (int, float)):
                products_with_ratings.append(product)
        
        if products_with_ratings:
            # Sort by rating
            top_rated = sorted(products_with_ratings, key=lambda x: x.get('rating', 0), reverse=True)[:3]
            
            st.write("#### " + t("top_rated"))
            for product in top_rated:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if product.get('images'):
                        st.image(product['images'][0], width=80)
                
                with col2:
                    st.write(f"**{product.get('name', 'Unknown')}**")
                    st.write(f"Rating: {product.get('rating', 'N/A')}/5")
                    st.write(f"Price: {product.get('price', 'N/A')}")
                    
                    # Get product recommendations
                    recommendations = content_analyzer.get_product_recommendations(product)
                    
                    if recommendations.get('similar_products') or recommendations.get('better_alternatives'):
                        with st.expander(t("view_similar")):
                            # Show similar products
                            if recommendations.get('similar_products'):
                                st.write("##### " + t("similar_products"))
                                for similar in recommendations['similar_products']:
                                    st.write(f"- {similar['product'].get('name', 'Unknown')} ({similar['similarity']:.0%} similar)")
                                    st.write(f"  Rating: {similar['product'].get('rating', 'N/A')}/5")
                                    st.write(f"  Price: {similar['product'].get('price', 'N/A')}")
                            
                            # Show better alternatives
                            if recommendations.get('better_alternatives'):
                                st.write("##### " + t("better_alternatives"))
                                for alternative in recommendations['better_alternatives']:
                                    st.write(f"- {alternative['product'].get('name', 'Unknown')}")
                                    st.write(f"  Rating: {alternative['product'].get('rating', 'N/A')}/5 (+{alternative['rating_improvement']:.1f})")
                                    st.write(f"  Price: {alternative['product'].get('price', 'N/A')}")

# New function to show detailed sentiment analysis
def show_sentiment_analysis(analysis):
    """Display detailed sentiment analysis"""
    if analysis.get('review_analysis'):
        st.write("### " + t("review_sentiment"))
        
        review_analysis = analysis['review_analysis']
        
        # Overall sentiment
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(t("overall_sentiment"), 
                     "Positive" if review_analysis['sentiment_counts']['positive'] > review_analysis['sentiment_counts']['negative'] else "Negative")
        with col2:
            st.metric(t("average_rating"), f"{review_analysis['average_rating']:.1f}/5")
        with col3:
            st.metric(t("reviews"), review_analysis['total_reviews'])
        
        # Sentiment distribution chart
        fig = px.pie(
            values=list(review_analysis['sentiment_counts'].values()),
            names=list(review_analysis['sentiment_counts'].keys()),
            title=t("sentiment_distribution")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Aspect analysis
        if review_analysis.get('aspect_counts'):
            st.write("#### " + t("review_analysis"))
            aspects_df = pd.DataFrame({
                'Aspect': list(review_analysis['aspect_counts'].keys()),
                'Mentions': list(review_analysis['aspect_counts'].values())
            })
            aspects_df = aspects_df.sort_values('Mentions', ascending=False)
            
            fig = px.bar(aspects_df, x='Aspect', y='Mentions', 
                        title="Most Discussed Product Aspects")
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample reviews
        st.write("#### Sample Reviews")
        sample_reviews = review_analysis['individual_analyses'][:3]  # Show first 3
        for review, sentiment, rating in sample_reviews:
            sentiment_color = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'gray'
            }.get(sentiment, 'gray')
            
            st.markdown(f"""
            <div style="border-left: 4px solid {sentiment_color}; padding-left: 10px; margin: 10px 0;">
                <p>{review}</p>
                <p><strong>{t('sentiment')}:</strong> <span style="color: {sentiment_color}">{sentiment}</span> | 
                <strong>{t('rating')}:</strong> {rating}/5</p>
            </div>
            """, unsafe_allow_html=True)

def signup():
    st.title(t("signup_title"))
    with st.form("signup_form"):
        email = st.text_input(t("email"))
        password = st.text_input(t("password"), type="password")
        submitted = st.form_submit_button(t("submit_signup"))
        
        if submitted:
            if not email or not password:
                st.error(t("fill_all_fields"))
                return
                
            try:
                with open("users.json", "r") as f:
                    data = json.load(f)
                
                if any(user["email"] == email for user in data["users"]):
                    st.error(t("email_exists"))
                    return
                
                data["users"].append({
                    "email": email,
                    "password": password
                })
                
                with open("users.json", "w") as f:
                    json.dump(data, f)
                
                st.success(t("account_created"))
                st.session_state.show_login = True
                
            except Exception as e:
                st.error(f"Error creating account: {str(e)}")

def login():
    st.title(t("login_title"))
    with st.form("login_form"):
        email = st.text_input(t("email"))
        password = st.text_input(t("password"), type="password")
        submitted = st.form_submit_button(t("submit_login"))
        
        if submitted:
            try:
                with open("users.json", "r") as f:
                    data = json.load(f)
                
                for user in data["users"]:
                    if user["email"] == email and user["password"] == password:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user
                        st.success(t("login_success"))
                        return
                
                st.error(t("invalid_credentials"))
                
            except Exception as e:
                st.error(f"Error during login: {str(e)}")

# Enhanced main function with new features
def main():
    initialize_database()
    initialize_user_history()
    
    # Language selection in sidebar
    st.sidebar.title(t("language_select"))
    language = st.sidebar.selectbox("", list(translations.keys()), index=list(translations.keys()).index(st.session_state.language))
    if language != st.session_state.language:
        st.session_state.language = language
        st.rerun()
    
    if not st.session_state.logged_in:
        st.sidebar.title(t("auth"))
        if st.sidebar.button(t("login_btn")):
            st.session_state.show_login = True
        if st.sidebar.button(t("signup_btn")):
            st.session_state.show_signup = True
            
        if st.session_state.get("show_login"):
            login()
        elif st.session_state.get("show_signup"):
            signup()
        return
    if "current_page" not in st.session_state:
        st.session_state.current_page = t("home")
    
    st.sidebar.title(t("navigation"))
    page = st.sidebar.radio(
        "", 
        [t("home"), t("analysis"), t("comparison"), t("dashboard"), t("project_suggestions")],
        index=[t("home"), t("analysis"), t("comparison"), t("dashboard"), t("project_suggestions")].index(st.session_state.current_page)
    )
    
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.rerun()

    if st.session_state.current_page == t("home"):
        st.title(t("title"))
        st.write(t("welcome"))
        
    elif st.session_state.current_page == t("analysis"):
        st.title(t("analysis"))
        url = st.text_input(t("enter_url"))

        if hasattr(st.session_state, 'prefill_url'):
            url = st.session_state.prefill_url
            del st.session_state.prefill_url
        
        if url:
            if st.button(t("start_scraping")):
                with st.spinner(t("scraping_analyzing")):
                    try:
                        scraper = WebScraper()
                        analyzer = ContentAnalyzer()
                        
                        content = scraper.scrape_website(url)
                        if not content or not content['text']:
                            st.error(t("scraping_failed"))
                            return
                            
                        chunks = analyzer.process_text(content['text'])
                        if not chunks:
                            st.error(t("no_content"))
                            return
                            
                        analysis = analyzer.analyze_content(content, url=url)
                        
                        # Store in history
                        st.session_state.analysis_history.append(analysis)
                        
                        # Display results
                        visualize_analysis(analysis)
                        
                        # Show sentiment analysis
                        show_sentiment_analysis(analysis)
                        
                        # Show similar projects
                        show_similar_projects(analysis, url)
                        
                        # Show product recommendations
                        show_product_recommendations(analysis)
                        
                        # Download options
                        st.subheader(t("download_results"))
                        
                        # Store analysis results in session state
                        st.session_state.analysis_results = analysis
                        
                        # Create download buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            html_content = download_as_html(analysis)
                            st.download_button(
                                label=t("download_html"),
                                data=html_content,
                                file_name="analysis_results.html",
                                mime="text/html"
                            )
                                
                    except Exception as e:
                        logger.error(f"Error during analysis: {e}")
                        st.error(t("analysis_error"))
        else:
            st.info("Please enter a URL to analyze.")
            
    elif st.session_state.current_page == t("comparison"):
        st.title(t("comparison"))
        url1 = st.text_input(t("enter_first_url"))
        url2 = st.text_input(t("enter_second_url"))
        question = st.text_input(t("comparison_question"), 
                               placeholder=t("question_placeholder"))
        if hasattr(st.session_state, 'compare_url1') and hasattr(st.session_state, 'compare_url2'):
            url1 = st.session_state.compare_url1
            url2 = st.session_state.compare_url2
            del st.session_state.compare_url1
            del st.session_state.compare_url2

        if url1 and url2:
            if st.button(t("start_comparison")):
                with st.spinner(t("scraping_analyzing")):
                    try:
                        scraper = WebScraper()
                        analyzer = ContentAnalyzer()
                        
                        # Scrape first website
                        content1 = scraper.scrape_website(url1)
                        if not content1 or not content1['text']:
                            st.error(t("scraping_first").format(url1))
                            return
                            
                        # Scrape second website
                        content2 = scraper.scrape_website(url2)
                        if not content2 or not content2['text']:
                            st.error(t("scraping_second").format(url2))
                            return
                        
                        # Process content
                        chunks1 = analyzer.process_text(content1['text'])
                        chunks2 = analyzer.process_text(content2['text'])
                        
                        if not chunks1 or not chunks2:
                            st.error(t("no_content"))
                            return
                        
                        # Analyze content
                        analysis1 = analyzer.analyze_content(content1, url=url1)
                        analysis2 = analyzer.analyze_content(content2, url=url2)
                        
                        # Store in history
                        st.session_state.comparison_history.append({
                            'url1': url1,
                            'url2': url2,
                            'analysis1': analysis1,
                            'analysis2': analysis2
                        })
                        
                        # Compare websites
                        comparison_results = analyzer.compare_websites(analysis1, analysis2, question)
                        

                        # Store comparison results in session state
                        st.session_state.comparison_results = comparison_results
                        
                        # Download as HTML
                        def gendownload_as_html(data):
                            if isinstance(data, dict):
                                html_parts = ["<html><head><title>Comparison Results</title></head><body>"]
                                html_parts.append("<h2>Website Comparison</h2>")
                                for section, content in data.items():
                                    html_parts.append(f"<h3>{section}</h3>")
                                    html_parts.append(f"<p>{content}</p>")
                                html_parts.append("</body></html>")
                                return "\n".join(html_parts)
                            else:
                                return f"<html><body><h2>Comparison Results</h2><p>{str(data)}</p></body></html>"
                    
                        # Download options
                        st.subheader(t("download_results"))
                        html_data = gendownload_as_html(comparison_results)
                        st.download_button(
                            label=t("download_html"),
                            data=html_data,
                            file_name="comparison_results.html",
                            mime="text/html"
                        )
                                
                    except Exception as e:
                        logger.error(f"Error during comparison: {e}")
                        st.error(t("comparison_error").format(str(e)))

        else:
            st.info(t("enter_both_urls"))
                
    elif st.session_state.current_page == t("dashboard"):
        st.title(t("user_dashboard"))
        st.write(t("welcome_user").format(st.session_state.user_info['email']))
        
        # Dashboard Statistics
        st.subheader("📊 Dashboard Overview")
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            analysis_count = len(st.session_state.analysis_history)
            st.metric("Total Analyses", analysis_count)
        
        with col2:
            comparison_count = len(st.session_state.comparison_history)
            st.metric("Total Comparisons", comparison_count)
        
        with col3:
            total_products = sum(len(analysis.get('products', [])) for analysis in st.session_state.analysis_history)
            st.metric("Products Analyzed", total_products)
        
        with col4:
            total_reviews = sum(analysis.get('statistics', {}).get('total_reviews', 0) for analysis in st.session_state.analysis_history)
            st.metric("Reviews Processed", total_reviews)
        
        # Recent Activity Section
        st.subheader("🕒 Recent Activity")
        
        if st.session_state.analysis_history:
            # Show last 5 analyses
            recent_analyses = st.session_state.analysis_history[-5:][::-1]  # Show latest first
            
            for i, analysis in enumerate(recent_analyses):
                with st.expander(f"Analysis {len(st.session_state.analysis_history)-i}: {analysis.get('url', 'Unknown URL')}", expanded=i==0):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        products_count = len(analysis.get('products', []))
                        st.metric("Products", products_count)
                    
                    with col2:
                        categories_count = len(analysis.get('categories', []))
                        st.metric("Categories", categories_count)
                    
                    with col3:
                        reviews_count = analysis.get('statistics', {}).get('total_reviews', 0)
                        st.metric("Reviews", reviews_count)
                    
                    # Quick summary
                    st.write("**Summary:**")
                    summary = analysis.get('summary', 'No summary available.')
                    st.write(summary[:200] + "..." if len(summary) > 200 else summary)
                    
                    # Sentiment overview if available
                    if analysis.get('review_analysis'):
                        sentiment = analysis['review_analysis']
                        st.write("**Sentiment Overview:**")
                        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
                        
                        with sentiment_col1:
                            positive_pct = (sentiment['sentiment_counts']['positive'] / sentiment['total_reviews']) * 100
                            st.metric("Positive", f"{positive_pct:.1f}%")
                        
                        with sentiment_col2:
                            negative_pct = (sentiment['sentiment_counts']['negative'] / sentiment['total_reviews']) * 100
                            st.metric("Negative", f"{negative_pct:.1f}%")
                        
                        with sentiment_col3:
                            st.metric("Avg Rating", f"{sentiment['average_rating']:.1f}/5")
        else:
            st.info("No analysis history yet. Start analyzing websites to see your activity here.")
        
        # Comparison History
        if st.session_state.comparison_history:
            st.subheader("⚖️ Recent Comparisons")
            recent_comparisons = st.session_state.comparison_history[-3:][::-1]
            
            for i, comparison in enumerate(recent_comparisons):
                with st.expander(f"Comparison {len(st.session_state.comparison_history)-i}: {comparison['url1']} vs {comparison['url2']}"):
                    analysis1 = comparison['analysis1']
                    analysis2 = comparison['analysis2']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**{comparison['url1']}**")
                        st.write(f"Products: {len(analysis1.get('products', []))}")
                        st.write(f"Categories: {len(analysis1.get('categories', []))}")
                        if analysis1.get('review_analysis'):
                            st.write(f"Avg Rating: {analysis1['review_analysis']['average_rating']:.1f}/5")
                    
                    with col2:
                        st.write(f"**{comparison['url2']}**")
                        st.write(f"Products: {len(analysis2.get('products', []))}")
                        st.write(f"Categories: {len(analysis2.get('categories', []))}")
                        if analysis2.get('review_analysis'):
                            st.write(f"Avg Rating: {analysis2['review_analysis']['average_rating']:.1f}/5")
        
        # User Insights and Recommendations
        st.subheader("💡 Your Insights")
        
        if st.session_state.analysis_history:
            # Most analyzed categories
            all_categories = []
            for analysis in st.session_state.analysis_history:
                all_categories.extend(analysis.get('categories', []))
            
            if all_categories:
                from collections import Counter
                category_counts = Counter(all_categories)
                top_categories = category_counts.most_common(5)
                
                st.write("**Your Top Categories:**")
                for category, count in top_categories:
                    st.write(f"• {category}: {count} analyses")
            
            # Product type analysis
            all_products = []
            for analysis in st.session_state.analysis_history:
                all_products.extend(analysis.get('products', []))
            
            if all_products:
                # Price range analysis
                prices = []
                for product in all_products:
                    if product.get('price'):
                        try:
                            price_str = str(product['price']).replace('₹', '').replace(',', '').strip()
                            price = float(price_str)
                            prices.append(price)
                        except:
                            continue
                
                if prices:
                    avg_price = sum(prices) / len(prices)
                    st.metric("Average Product Price", f"₹{avg_price:,.2f}")
                
                # Brand analysis
                brands = [p.get('brand') for p in all_products if p.get('brand')]
                if brands:
                    brand_counts = Counter(brands)
                    top_brand = brand_counts.most_common(1)[0]
                    st.write(f"**Most Analyzed Brand:** {top_brand[0]} ({top_brand[1]} products)")
        
        # Achievement System
        st.subheader("🏆 Achievements")
        
        achievements = []
        
        # Analysis achievements
        if analysis_count >= 1:
            achievements.append("🔍 First Analysis - Completed your first website analysis")
        if analysis_count >= 5:
            achievements.append("📊 Analysis Pro - Completed 5 analyses")
        if analysis_count >= 10:
            achievements.append("🔬 Research Expert - Completed 10 analyses")
        
        # Comparison achievements
        if comparison_count >= 1:
            achievements.append("⚖️ Comparator - Completed your first comparison")
        if comparison_count >= 3:
            achievements.append("📈 Comparison Master - Completed 3 comparisons")
        
        # Product achievements
        if total_products >= 10:
            achievements.append("🛍️ Product Explorer - Analyzed 10+ products")
        if total_products >= 50:
            achievements.append("📦 Product Guru - Analyzed 50+ products")
        
        # Review achievements
        if total_reviews >= 10:
            achievements.append("💬 Review Reader - Processed 10+ reviews")
        if total_reviews >= 100:
            achievements.append("🎯 Sentiment Analyst - Processed 100+ reviews")
        
        if achievements:
            for achievement in achievements:
                st.success(achievement)
        else:
            st.info("Complete more analyses to unlock achievements!")
        
        # Quick Actions - FIXED NAVIGATION
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 New Analysis", use_container_width=True):
                # Set the page to analysis and rerun
                st.session_state.current_page = t("analysis")
                st.rerun()
        
        with col2:
            if st.button("⚖️ New Comparison", use_container_width=True):
                # Set the page to comparison and rerun
                st.session_state.current_page = t("comparison")
                st.rerun()
        
        with col3:
            if st.button("📋 View All History", use_container_width=True):
                # Set the page to project suggestions and rerun
                st.session_state.page = t("project_suggestions")
                st.rerun()
        
        # Data Export
        st.subheader("💾 Export Data")
        
        if st.session_state.analysis_history:
            # Create comprehensive export data
            export_data = {
                "user": st.session_state.user_info['email'],
                "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analyses": st.session_state.analysis_history,
                "comparisons": st.session_state.comparison_history,
                "statistics": {
                    "total_analyses": analysis_count,
                    "total_comparisons": comparison_count,
                    "total_products": total_products,
                    "total_reviews": total_reviews
                }
            }
            
            export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Export All Data (JSON)",
                data=export_json,
                file_name=f"web_analysis_export_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Logout button at the bottom
        st.markdown("---")
        if st.button(t("logout"), use_container_width=True, type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.rerun()

        
    elif st.session_state.current_page == t("project_suggestions"):
        st.title(t("project_suggestions"))
        
        if st.session_state.analysis_history:
            # Create tabs for different types of suggestions
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Insights", "🛍️ Product Recommendations", "⚖️ Comparison Ideas", "📈 Trends & Patterns"])
            
            with tab1:
                st.subheader("📊 Your Analysis Insights")
                
                # Overall statistics
                total_analyses = len(st.session_state.analysis_history)
                total_products = sum(len(analysis.get('products', [])) for analysis in st.session_state.analysis_history)
                total_reviews = sum(analysis.get('statistics', {}).get('total_reviews', 0) for analysis in st.session_state.analysis_history)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analyses", total_analyses)
                with col2:
                    st.metric("Products Analyzed", total_products)
                with col3:
                    st.metric("Reviews Processed", total_reviews)
                
                # Most analyzed categories
                all_categories = []
                for analysis in st.session_state.analysis_history:
                    all_categories.extend(analysis.get('categories', []))
                
                if all_categories:
                    from collections import Counter
                    category_counts = Counter(all_categories)
                    top_categories = category_counts.most_common(10)
                    
                    st.write("#### 🏷️ Your Top Categories")
                    category_data = pd.DataFrame(top_categories, columns=['Category', 'Count'])
                    fig = px.bar(category_data, x='Category', y='Count', 
                            title="Most Frequently Analyzed Categories",
                            color='Count', color_continuous_scale='blues')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Analysis timeline
                st.write("#### 📅 Analysis Timeline")
                analysis_dates = [f"Analysis {i+1}" for i in range(total_analyses)]
                product_counts = [len(analysis.get('products', [])) for analysis in st.session_state.analysis_history]
                
                timeline_df = pd.DataFrame({
                    'Analysis': analysis_dates,
                    'Products Found': product_counts
                })
                
                fig = px.line(timeline_df, x='Analysis', y='Products Found', 
                            title="Product Discovery Over Time", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("🛍️ Smart Product Recommendations")
                
                all_products = []
                for analysis in st.session_state.analysis_history:
                    for product in analysis.get('products', []):
                        product['source_url'] = analysis.get('url', 'Unknown')
                        all_products.append(product)
                
                if all_products:
                    # Top rated products
                    rated_products = [p for p in all_products if p.get('rating') and isinstance(p.get('rating'), (int, float))]
                    if rated_products:
                        top_rated = sorted(rated_products, key=lambda x: x.get('rating', 0), reverse=True)[:10]
                        
                        st.write("#### ⭐ Top Rated Products")
                        for i, product in enumerate(top_rated[:5]):
                            with st.container():
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    if product.get('images'):
                                        st.image(product['images'][0], width=80)
                                    else:
                                        st.image("https://via.placeholder.com/80x80?text=No+Image", width=80)
                                
                                with col2:
                                    st.write(f"**{product.get('name', 'Unknown Product')}**")
                                    st.write(f"**Rating:** {product.get('rating')}/5 | **Price:** {product.get('price', 'N/A')}")
                                    st.write(f"**Brand:** {product.get('brand', 'Unknown')} | **Category:** {product.get('category', 'Unknown')}")
                                    st.write(f"*Source:* {product.get('source_url', 'Unknown')}")
                                
                                st.markdown("---")
                    
                    # Price-based recommendations
                    priced_products = [p for p in all_products if p.get('price')]
                    if priced_products:
                        # Budget picks (lowest priced)
                        try:
                            budget_products = sorted(priced_products, 
                                                key=lambda x: float(str(x.get('price', '0')).replace('₹', '').replace(',', '').strip() or '0'))[:5]
                            
                            st.write("#### 💰 Budget-Friendly Picks")
                            for product in budget_products:
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    if product.get('images'):
                                        st.image(product['images'][0], width=60)
                                with col2:
                                    st.write(f"**{product.get('name', 'Unknown')}** - {product.get('price')}")
                        except:
                            pass
                    
                    # Category-based recommendations
                    if all_products:
                        st.write("#### 🎯 Recommendations by Category")
                        categories = set(p.get('category') for p in all_products if p.get('category'))
                        
                        for category in list(categories)[:5]:  # Show top 5 categories
                            category_products = [p for p in all_products if p.get('category') == category]
                            if category_products:
                                with st.expander(f"📁 {category} ({len(category_products)} products)"):
                                    for product in category_products[:3]:
                                        st.write(f"• **{product.get('name', 'Unknown')}** - {product.get('price', 'Price N/A')}")
            
            with tab3:
                st.subheader("⚖️ Smart Comparison Ideas")
                
                # Generate comparison suggestions based on your analysis history
                comparison_suggestions = []
                
                # Suggest comparisons between different analyses
                if len(st.session_state.analysis_history) >= 2:
                    st.write("#### 🔄 Compare Your Analyses")
                    
                    for i in range(min(3, len(st.session_state.analysis_history))):
                        analysis1 = st.session_state.analysis_history[-(i+1)]
                        analysis2 = st.session_state.analysis_history[-(i+2)] if i+2 <= len(st.session_state.analysis_history) else st.session_state.analysis_history[0]
                        
                        url1 = analysis1.get('url', 'Unknown URL 1')
                        url2 = analysis2.get('url', 'Unknown URL 2')
                        
                        products1 = len(analysis1.get('products', []))
                        products2 = len(analysis2.get('products', []))
                        
                        with st.container():
                            st.write(f"**Comparison Suggestion {i+1}**")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**{url1}**")
                                st.write(f"• Products: {products1}")
                                st.write(f"• Categories: {len(analysis1.get('categories', []))}")
                                if analysis1.get('review_analysis'):
                                    st.write(f"• Avg Rating: {analysis1['review_analysis']['average_rating']:.1f}/5")
                            
                            with col2:
                                st.write(f"**{url2}**")
                                st.write(f"• Products: {products2}")
                                st.write(f"• Categories: {len(analysis2.get('categories', []))}")
                                if analysis2.get('review_analysis'):
                                    st.write(f"• Avg Rating: {analysis2['review_analysis']['average_rating']:.1f}/5")
                            
                            if st.button(f"Compare These Websites", key=f"compare_{i}"):
                                # Set the URLs and navigate to comparison page
                                st.session_state.compare_url1 = url1
                                st.session_state.compare_url2 = url2
                                st.session_state.current_page = t("comparison")
                                st.rerun()
                            
                            st.markdown("---")
                
                # Product comparison suggestions
                all_products = []
                for analysis in st.session_state.analysis_history:
                    for product in analysis.get('products', []):
                        product['source_url'] = analysis.get('url', 'Unknown')
                        all_products.append(product)
                
                if all_products:
                    # Find similar products for comparison
                    st.write("#### 🔍 Compare Similar Products")
                    
                    # Group by category
                    products_by_category = {}
                    for product in all_products:
                        category = product.get('category', 'Uncategorized')
                        if category not in products_by_category:
                            products_by_category[category] = []
                        products_by_category[category].append(product)
                    
                    # Show comparison suggestions for each category
                    for category, products in list(products_by_category.items())[:3]:  # Top 3 categories
                        if len(products) >= 2:
                            with st.expander(f"🔄 Compare {category} Products"):
                                # Show top products in this category
                                for i in range(min(3, len(products))):
                                    product = products[i]
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                    with col1:
                                        st.write(f"**{product.get('name', 'Unknown')}**")
                                    with col2:
                                        st.write(f"Price: {product.get('price', 'N/A')}")
                                    with col3:
                                        st.write(f"Rating: {product.get('rating', 'N/A')}")
            
            with tab4:
                st.subheader("📈 Market Trends & Patterns")
                
                all_products = []
                for analysis in st.session_state.analysis_history:
                    for product in analysis.get('products', []):
                        product['source_url'] = analysis.get('url', 'Unknown')
                        all_products.append(product)
                
                if all_products:
                    # Price distribution analysis
                    prices = []
                    for product in all_products:
                        if product.get('price'):
                            try:
                                price_str = str(product['price']).replace('₹', '').replace(',', '').strip()
                                price = float(price_str)
                                prices.append(price)
                            except:
                                continue
                    
                    if prices:
                        st.write("#### 💸 Price Distribution Analysis")
                        price_df = pd.DataFrame(prices, columns=['Price'])
                        fig = px.histogram(price_df, x='Price', title="Product Price Distribution",
                                        nbins=20, color_discrete_sequence=['#00CC96'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        avg_price = sum(prices) / len(prices)
                        min_price = min(prices)
                        max_price = max(prices)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Price", f"₹{avg_price:,.2f}")
                        with col2:
                            st.metric("Minimum Price", f"₹{min_price:,.2f}")
                        with col3:
                            st.metric("Maximum Price", f"₹{max_price:,.2f}")
                    
                    # Brand analysis
                    brands = [p.get('brand') for p in all_products if p.get('brand')]
                    if brands:
                        brand_counts = Counter(brands)
                        top_brands = brand_counts.most_common(10)
                        
                        st.write("#### 🏢 Top Brands in Your Analysis")
                        brand_df = pd.DataFrame(top_brands, columns=['Brand', 'Count'])
                        fig = px.pie(brand_df, values='Count', names='Brand', 
                                title="Brand Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Category price comparison
                    category_prices = {}
                    for product in all_products:
                        category = product.get('category')
                        price = product.get('price')
                        if category and price:
                            try:
                                price_val = float(str(price).replace('₹', '').replace(',', '').strip())
                                if category not in category_prices:
                                    category_prices[category] = []
                                category_prices[category].append(price_val)
                            except:
                                continue
                    
                    if category_prices:
                        st.write("#### 📊 Average Prices by Category")
                        avg_prices = {cat: sum(prices)/len(prices) for cat, prices in category_prices.items() if prices}
                        if avg_prices:
                            avg_price_df = pd.DataFrame(list(avg_prices.items()), columns=['Category', 'Average Price'])
                            avg_price_df = avg_price_df.sort_values('Average Price', ascending=False)
                            
                            fig = px.bar(avg_price_df, x='Category', y='Average Price',
                                    title="Average Product Price by Category",
                                    color='Average Price', color_continuous_scale='viridis')
                            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("🎯 No analysis history yet. Here's what you can do:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**1. Start Analyzing**")
                st.write("Analyze e-commerce websites to discover products, prices, and reviews")
                if st.button("Begin Analysis", key="suggest_analysis"):
                    # Navigate to analysis page
                    st.session_state.current_page = t("analysis")
                    st.rerun()
            
            with col2:
                st.write("**2. Compare Websites**")
                st.write("Compare different websites to find the best products and prices")
                if st.button("Start Comparison", key="suggest_comparison"):
                    # Navigate to comparison page
                    st.session_state.current_page = t("comparison")
                    st.rerun()
            
            with col3:
                st.write("**3. Get Insights**")
                st.write("Once you have data, we'll provide personalized recommendations and insights")
            
            st.markdown("---")
            st.write("### 💡 Suggested Starting Points:")
            st.write("• Analyze popular e-commerce sites like Amazon, Flipkart, or Myntra")
            st.write("• Compare prices for the same product across different websites")
            st.write("• Look for products with high ratings and positive reviews")
            st.write("• Track price trends for products you're interested in")

# Global analyzer instance
content_analyzer = ContentAnalyzer()

# Run the application
if __name__ == "__main__":
    main()