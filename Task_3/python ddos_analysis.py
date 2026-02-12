import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def parse_log_file(file_path):
    """Parses the web server log file to extract timestamps."""
    log_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})\]')
    timestamps = []

    print(f"Reading log file: {file_path}...")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    timestamps.append(match.group(1))
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()

    print(f"Parsed {len(timestamps)} log entries.")
    return pd.DataFrame({'timestamp': pd.to_datetime(timestamps)})


def analyze_ddos_traffic(log_file):
    df = parse_log_file(log_file)
    if df.empty:
        return

    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # UPDATED: Using lowercase '1s' for compatibility with newer pandas versions
    traffic_data = df.resample('1s').size().reset_index(name='requests')

    start_time = traffic_data['timestamp'].min()
    traffic_data['seconds_from_start'] = (traffic_data['timestamp'] - start_time).dt.total_seconds()

    X = traffic_data[['seconds_from_start']]
    y = traffic_data['requests']

    model = LinearRegression()
    model.fit(X, y)
    traffic_data['predicted_requests'] = model.predict(X)

    traffic_data['residuals'] = traffic_data['requests'] - traffic_data['predicted_requests']
    std_dev = traffic_data['residuals'].std()

    threshold_multiplier = 2
    traffic_data['threshold'] = traffic_data['predicted_requests'] + (threshold_multiplier * std_dev)
    traffic_data['is_attack'] = traffic_data['requests'] > traffic_data['threshold']

    attack_points = traffic_data[traffic_data['is_attack']].copy()

    print("\n" + "=" * 40)
    print("DDoS Attack Detection Report")
    print("=" * 40)

    if not attack_points.empty:
        attack_points['group_id'] = (attack_points['seconds_from_start'].diff() > 10).cumsum()
        for _, group in attack_points.groupby('group_id'):
            start = group['timestamp'].min()
            end = group['timestamp'].max()
            peak_rps = group['requests'].max()
            duration = (end - start).total_seconds()
            print(
                f"Attack Detected!\n  Start Time: {start}\n  End Time:   {end}\n  Duration:   {duration:.0f} seconds\n  Peak RPS:   {peak_rps}\n" + "-" * 40)
    else:
        print("No significant anomalies detected.")

    plt.figure(figsize=(14, 7))
    plt.plot(traffic_data['timestamp'], traffic_data['requests'], label='Actual Requests/Sec', color='blue', alpha=0.6)
    plt.plot(traffic_data['timestamp'], traffic_data['predicted_requests'], label='Regression Line', color='red',
             linestyle='--')
    plt.plot(traffic_data['timestamp'], traffic_data['threshold'], label='Threshold', color='green', linestyle=':')

    if not attack_points.empty:
        plt.scatter(attack_points['timestamp'], attack_points['requests'], color='orange', label='Anomaly', s=30,
                    zorder=5)

    plt.title('Web Server Traffic Analysis: DDoS Detection')
    plt.xlabel('Time')
    plt.ylabel('Requests per Second')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ddos_regression_plot.png')
    plt.show()


if __name__ == "__main__":
    LOG_FILENAME = 'i_iakashvili25_71384_server.log'
    analyze_ddos_traffic(LOG_FILENAME)