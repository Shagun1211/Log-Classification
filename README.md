🧾 Log Classification System

A machine learning–based log classification system that automatically categorizes log messages using a hybrid approach combining regex rules, BERT-based NLP, and LLM-based classification for legacy logs.

This project is designed to handle real-world system/application logs efficiently and intelligently.

🚀 Features

🔍 Regex-based classification for fast and deterministic log parsing

🧠 BERT-based text classification for unstructured logs

🤖 LLM-based classification for complex or legacy CRM logs

⚡ Modular and extensible architecture

🌐 API support using FastAPI

📊 Training pipeline with synthetic log generation

Project Structure

Log-Classification/
│
├── classify.py                  
├── main.py                     
├── server.py                    
├── processor_regex.py           
├── processor_bert.py           
├── processor_llm.py             
├── requirements.txt             
├── .gitignore                   
│
├── models/
│   └── log_classifier_model.joblib
│
├── training/
│   ├── dataset/
│   │   └── synthetic_logs.csv
│   └── training.ipynb
│
├── resources/
│   └── test.csv


Classification Strategy

The system uses a tiered decision approach:

Regex Classifier

Fast and rule-based

Used for well-defined log patterns

BERT Classifier

Handles general unstructured text

Applied when regex fails

LLM Classifier

Used for complex or legacy CRM logs

Provides semantic understanding

Technologies Used

Python

FastAPI

Scikit-learn

BERT (Transformers)

Regex

Joblib

Pandas, NumPy

Future Improvements

Add real-time log streaming (Kafka / RabbitMQ)

Improve model accuracy with larger datasets

Add dashboard for visualization

Implement Git LFS for large model files
