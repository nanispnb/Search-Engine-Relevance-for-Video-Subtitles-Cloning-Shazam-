import streamlit as st
import speech_recognition as sr
import sqlite3
import zlib
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma("subtitle_search_db", embedding_function=embedding_model)

def extract_subtitles(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT num, content FROM zipfiles")
    rows = cursor.fetchall()
    subtitles = []
    for num, content in rows:
        try:
            subtitle_text = zlib.decompress(content).decode('latin-1')
            subtitles.append((num, subtitle_text))
        except:
            continue
    conn.close()
    return subtitles

db_path = "E:\Innomatics\eng_subtitles_database.db" 
subtitles = extract_subtitles(db_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
for num, subtitle_text in subtitles:
    chunks = text_splitter.split_text(subtitle_text)
    embeddings = embedding_model.embed_documents(chunks)
    for chunk, embedding in zip(chunks, embeddings):
        vector_store.add_texts([chunk], metadatas=[{"num": num}])

st.title("Video Subtitle Search Engine")

uploaded_audio = st.file_uploader("Upload a 2-minute audio clip", type=["wav", "mp3"])
if uploaded_audio:
    recognizer = sr.Recognizer()
    query_text = None 

    with sr.AudioFile(uploaded_audio) as source:
        audio_data = recognizer.record(source)
        try:
            query_text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            st.write("Could not understand the audio.")
        except sr.RequestError:
            st.write("Error with speech recognition service.")

    if query_text:  
        query_embedding = embedding_model.embed_query(query_text)
        results = vector_store.similarity_search_by_vector(query_embedding, k=5)

        st.write("## Top Matching Subtitle Segments")
        for idx, result in enumerate(results):
            st.write(f"**Result {idx + 1}:** {result.page_content}")

        st.write(f"## Transcribed Query: {query_text}")
    else:
        st.write("No valid transcription available for searching.")
