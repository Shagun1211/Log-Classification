from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import os

from processor_regex import classify_with_regex
from processor_bert import classify_logs_with_bert
from processor_llm import classify_with_llm

app = FastAPI(
    title="Log Classification API",
    description="Hybrid Log Classification using Regex, BERT, and LLM",
    version="1.0"
)


class LogRequest(BaseModel):
    source: str
    log_message: str

class LogResponse(BaseModel):
    source: str
    log_message: str
    target_label: str


def classify_log(source, log_message):

    label = classify_with_regex(log_message)

    if label != "Other":
        return label

    
    if source != "LegacyCRM":
        return classify_logs_with_bert([log_message])[0]

  
    return classify_with_llm(log_message)

@app.post("/classify-log", response_model=LogResponse)
def classify_single_log(request: LogRequest):
    label = classify_log(request.source, request.log_message)

    return {
        "source": request.source,
        "log_message": request.log_message,
        "target_label": label
    }


@app.post("/classify-csv")
def classify_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    required_cols = {"source", "log_message"}
    if not required_cols.issubset(df.columns):
        return {"error": "CSV must contain 'source' and 'log_message' columns"}

    labels = []
    for _, row in df.iterrows():
        label = classify_log(row["source"], row["log_message"])
        labels.append(label)

    df["target_label"] = labels

    output_path = "resources/output.csv"
    os.makedirs("resources", exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "message": "Classification completed",
        "rows": len(df),
        "output_file": output_path
    }

@app.get("/")
def health():
    return {"status": "Log Classification API running"}
