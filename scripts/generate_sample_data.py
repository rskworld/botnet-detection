"""
Sample Data Generator for Botnet Detection
Generates synthetic network traffic data for testing and demonstration

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

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(n_samples=1000, botnet_ratio=0.3, output_path='data/raw/network_traffic.csv'):
    """
    Generate synthetic network traffic data
    
    Args:
        n_samples: Number of samples to generate
        botnet_ratio: Ratio of botnet traffic (0.0 to 1.0)
        output_path: Path to save the generated data
    """
    np.random.seed(42)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_botnet = int(n_samples * botnet_ratio)
    n_normal = n_samples - n_botnet
    
    data = []
    
    # Generate normal traffic
    for i in range(n_normal):
        record = {
            'id': i,
            'duration': np.random.uniform(1, 300),  # seconds
            'packet_count': np.random.poisson(100),
            'total_bytes': np.random.uniform(1000, 100000),
            'connection_count': np.random.poisson(5),
            'dns_queries': np.random.poisson(2),
            'avg_packet_size': np.random.uniform(64, 1500),
            'unique_ips': np.random.poisson(3),
            'port_count': np.random.poisson(2),
            'is_botnet': 0,
            'label': 'normal'
        }
        data.append(record)
    
    # Generate botnet traffic (different patterns)
    for i in range(n_normal, n_samples):
        record = {
            'id': i,
            'duration': np.random.uniform(1, 600),  # Longer duration
            'packet_count': np.random.poisson(500),  # More packets
            'total_bytes': np.random.uniform(50000, 500000),  # More bytes
            'connection_count': np.random.poisson(50),  # Many connections
            'dns_queries': np.random.poisson(20),  # More DNS queries
            'avg_packet_size': np.random.uniform(100, 2000),
            'unique_ips': np.random.poisson(20),  # Many unique IPs
            'port_count': np.random.poisson(10),  # Many ports
            'is_botnet': 1,
            'label': 'botnet'
        }
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} samples ({n_normal} normal, {n_botnet} botnet)")
    print(f"Data saved to {output_path}")
    
    # Also save processed version
    processed_path = Path('data/processed/training_data.csv')
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Training data saved to {processed_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample network traffic data')
    parser.add_argument('--samples', '-n', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--botnet-ratio', '-r', type=float, default=0.3, help='Ratio of botnet traffic')
    parser.add_argument('--output', '-o', default='data/raw/network_traffic.csv', help='Output file path')
    
    args = parser.parse_args()
    
    generate_sample_data(
        n_samples=args.samples,
        botnet_ratio=args.botnet_ratio,
        output_path=args.output
    )

