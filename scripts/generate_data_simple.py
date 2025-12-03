"""
Simple Data Generator for Botnet Detection
Generates synthetic network traffic data without external dependencies

Project: Botnet Detection with Machine Learning
Category: ML Projects
Developer: RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in
"""

import csv
import random
from pathlib import Path


def generate_sample_data(n_samples=1000, botnet_ratio=0.3, output_path='data/raw/network_traffic.csv'):
    """
    Generate synthetic network traffic data
    
    Args:
        n_samples: Number of samples to generate
        botnet_ratio: Ratio of botnet traffic (0.0 to 1.0)
        output_path: Path to save the generated data
    """
    random.seed(42)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_botnet = int(n_samples * botnet_ratio)
    n_normal = n_samples - n_botnet
    
    # CSV headers
    headers = [
        'id', 'duration', 'packet_count', 'total_bytes', 'connection_count',
        'dns_queries', 'avg_packet_size', 'unique_ips', 'port_count',
        'is_botnet', 'label'
    ]
    
    # Generate data
    data = []
    
    # Generate normal traffic
    for i in range(n_normal):
        record = {
            'id': i,
            'duration': round(random.uniform(1, 300), 2),
            'packet_count': random.randint(10, 200),
            'total_bytes': random.randint(1000, 100000),
            'connection_count': random.randint(1, 10),
            'dns_queries': random.randint(0, 5),
            'avg_packet_size': random.randint(64, 1500),
            'unique_ips': random.randint(1, 5),
            'port_count': random.randint(1, 5),
            'is_botnet': 0,
            'label': 'normal'
        }
        data.append(record)
    
    # Generate botnet traffic (different patterns)
    for i in range(n_normal, n_samples):
        record = {
            'id': i,
            'duration': round(random.uniform(1, 600), 2),
            'packet_count': random.randint(200, 1000),
            'total_bytes': random.randint(50000, 500000),
            'connection_count': random.randint(20, 100),
            'dns_queries': random.randint(10, 50),
            'avg_packet_size': random.randint(100, 2000),
            'unique_ips': random.randint(10, 50),
            'port_count': random.randint(5, 20),
            'is_botnet': 1,
            'label': 'botnet'
        }
        data.append(record)
    
    # Shuffle data
    random.shuffle(data)
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Generated {len(data)} samples ({n_normal} normal, {n_botnet} botnet)")
    print(f"Data saved to {output_path}")
    
    # Also save processed version
    processed_path = Path('data/processed/training_data.csv')
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(processed_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Training data saved to {processed_path}")
    
    # Print statistics
    print("\nData Statistics:")
    print(f"  Total records: {len(data)}")
    print(f"  Normal traffic: {n_normal} ({n_normal/len(data)*100:.1f}%)")
    print(f"  Botnet traffic: {n_botnet} ({n_botnet/len(data)*100:.1f}%)")
    
    return data


if __name__ == "__main__":
    import sys
    
    # Default values
    n_samples = 1000
    botnet_ratio = 0.3
    output_path = 'data/raw/network_traffic.csv'
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])
    if len(sys.argv) > 2:
        botnet_ratio = float(sys.argv[2])
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    
    generate_sample_data(
        n_samples=n_samples,
        botnet_ratio=botnet_ratio,
        output_path=output_path
    )

