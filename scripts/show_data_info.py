"""
Show Data Information
Displays information about generated data

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
from pathlib import Path
from collections import Counter


def show_data_info(file_path='data/raw/network_traffic.csv'):
    """Display information about the generated data"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    print("="*60)
    print("DATA INFORMATION")
    print("="*60)
    print(f"\nFile: {file_path}")
    print(f"Total Records: {len(data)}")
    
    # Count labels
    labels = [row['label'] for row in data]
    label_counts = Counter(labels)
    
    print("\nClass Distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {label.capitalize()}: {count:4d} ({percentage:5.2f}%)")
    
    # Show sample statistics
    print("\nSample Records:")
    print("-" * 60)
    for i, row in enumerate(data[:5]):
        print(f"\nRecord {i+1}:")
        print(f"  ID: {row['id']}")
        print(f"  Label: {row['label']}")
        print(f"  Duration: {row['duration']} seconds")
        print(f"  Packets: {row['packet_count']}")
        print(f"  Total Bytes: {row['total_bytes']}")
        print(f"  Connections: {row['connection_count']}")
        print(f"  DNS Queries: {row['dns_queries']}")
    
    # Calculate averages
    print("\n" + "="*60)
    print("AVERAGE VALUES")
    print("="*60)
    
    numeric_fields = ['duration', 'packet_count', 'total_bytes', 'connection_count', 
                     'dns_queries', 'avg_packet_size', 'unique_ips', 'port_count']
    
    for field in numeric_fields:
        values = [float(row[field]) for row in data]
        avg = sum(values) / len(values)
        print(f"  {field.replace('_', ' ').title()}: {avg:.2f}")
    
    print("\n" + "="*60)
    print("Data is ready for training!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/network_traffic.csv'
    show_data_info(file_path)

