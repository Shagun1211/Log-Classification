from processor_regex import classify_with_regex
from processor_bert import classify_logs_with_bert
from processor_llm import classify_with_llm
import pandas as pd

def classify(logs):
    labels = []
    for source, log_message in logs:
        label = classify_log_message(source, log_message)
        labels.append(label)
    return labels

def classify_log_message(source, log_message):
    
    label = classify_with_regex(log_message)

  
    if label is None or label == "None" or label == "Other":
        
        if source == "LegacyCRM":
            label = classify_with_llm(log_message)
        
        else:
            label = classify_logs_with_bert([log_message])[0]

    return label


def classify_csv(input_file):
    df = pd.read_csv(input_file)

    logs = list(zip(df["source"], df["log_message"]))
    df["target_label"] = classify(logs)

    output_file = "output.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Classification complete. Saved to {output_file}")

if __name__ == "__main__":
    classify_csv("resources/test.csv")
