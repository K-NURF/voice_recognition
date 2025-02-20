import re
import json

# Define the log file path
log_file_path = "training_log_run_2.txt"

# Regex patterns
train_pattern = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Training Step: (?P<step>\d+)"
)
eval_patterns = {
    "eval_loss": re.compile(r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?eval_loss: (?P<eval_loss>[0-9.]+)"),
    "eval_wer": re.compile(r"eval_wer: (?P<eval_wer>[0-9.]+)"),
    "eval_runtime": re.compile(r"eval_runtime: (?P<eval_runtime>[0-9.]+)"),
    "samples_per_second": re.compile(r"eval_samples_per_second: (?P<samples_per_second>[0-9.]+)"),
    "steps_per_second": re.compile(r"eval_steps_per_second: (?P<steps_per_second>[0-9.]+)"),
    "epoch": re.compile(r"epoch: (?P<epoch>[0-9.]+)")
}

# Initialize lists to store data
training_metrics = []
evaluation_metrics = []
seen_timestamps = set()  # To track unique timestamps for evaluation metrics

# Process the log file
with open(log_file_path, "r") as file:
    current_eval = {}
    for line in file:
        # Match training logs
        train_match = train_pattern.search(line)
        if train_match:
            training_metrics.append({
                "timestamp": train_match.group("timestamp"),
                "step": int(train_match.group("step"))
            })
        
        # Match evaluation logs
        for key, pattern in eval_patterns.items():
            match = pattern.search(line)
            if match:
                if key == "eval_loss":
                    # Start a new evaluation block
                    timestamp = match.group("timestamp")
                    if timestamp in seen_timestamps:
                        continue  # Skip duplicate timestamps
                    if current_eval:
                        evaluation_metrics.append(current_eval)
                        seen_timestamps.add(current_eval["timestamp"])
                    current_eval = {"timestamp": timestamp, key: float(match.group(key))}
                    print("Timestamp:", timestamp)
                    print("Seen Timestamps:", seen_timestamps)
                elif current_eval:
                    # Add other metrics to the current evaluation block
                    current_eval[key] = float(match.group(key))
    
    # Append the last evaluation block if it's not a duplicate
    if current_eval and current_eval["timestamp"] not in seen_timestamps:
        evaluation_metrics.append(current_eval)
        seen_timestamps.add(current_eval["timestamp"])

# Combine into a single JSON object
output = {
    "training_metrics": training_metrics,
    "evaluation_metrics": evaluation_metrics
}

# Save to JSON file
output_file_path = "parsed_metrics_2.json"
with open(output_file_path, "w") as json_file:
    json.dump(output, json_file, indent=4)

print(f"Metrics successfully extracted and saved to {output_file_path}.")
