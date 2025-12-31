# AI_NIDS_Project

This project is an AI-based Network Intrusion Detection System (NIDS) that leverages machine learning to detect and classify network attacks. It uses a dataset of network traffic and provides a Streamlit-based web interface for interactive analysis and visualization.

## Features
- Data preprocessing and feature engineering
- Machine learning model training and evaluation
- Real-time and batch intrusion detection
- Visualizations using Seaborn and Matplotlib
- User-friendly Streamlit web interface

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- streamlit
- seaborn
- matplotlib

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run nids_main.py
   ```

## Dataset
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: Contains network traffic data for training and testing the NIDS.

## Repository Structure
- `nids_main.py`: Main application code
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: Dataset file
- `README.md`: Project documentation

## License
This project is licensed under the MIT License.
