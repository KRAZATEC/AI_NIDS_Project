
# 🚦 AI_NIDS_Project: AI-based Network Intrusion Detection System 🚦

Welcome to the **AI_NIDS_Project**! This project leverages the power of Artificial Intelligence and Machine Learning to detect and classify network intrusions in real-time. Whether you're a cybersecurity enthusiast, data scientist, or network admin, this tool is designed to help you analyze, visualize, and secure your network traffic with ease. 🛡️

## ✨ Features
- 🔍 **Data Preprocessing & Feature Engineering**: Clean and transform raw network data, removing infinity values and handling missing data.
- 🤖 **Random Forest Classifier**: Train a powerful ML model to detect network attacks and benign traffic.
- 🎲 **Live Packet Simulation**: Capture random packets from test data to simulate real network traffic analysis.
- 🖥️ **Streamlit Web App**: User-friendly interactive interface for intrusion detection.
- 🤖 **AI-Powered Explanations**: Integrate with Groq AI (Llama 3.3) to generate detailed cybersecurity analysis of detected packets.
- 📈 **Real-time Detection**: Get instant predictions and explanations for network traffic classification.

## 📂 Project Structure
```
├── nids_main.py                # Main application code
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv  # Network traffic dataset
├── README.md                   # Project documentation
```

## 🗃️ Dataset
- **Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv**: Real-world network traffic data for training and testing the NIDS.

## 🚀 Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/KRAZATEC/AI_NIDS_Project.git
   cd AI_NIDS_Project
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or install manually:
   pip install pandas numpy scikit-learn streamlit groq
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run nids_main.py
   ```

## 🛠️ Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- streamlit
- groq (for AI-powered packet analysis)

## 💡 How It Works
1. **Load Data**: Import the CSV dataset containing network traffic features.
2. **Train Model**: Click "Train Model Now" to train a Random Forest classifier on the network traffic data.
3. **Simulate Traffic**: Use the "Capture Random Packet" button to select a random packet from the test data.
4. **Real-time Detection**: The model instantly predicts whether the packet is BENIGN or an ATTACK.
5. **AI Analysis**: Enter your Groq API key and click "Generate Explanation" to get detailed cybersecurity insights about the packet from Groq's Llama 3.3 AI model.

## 🔑 API Setup
To use the AI-powered packet analysis feature:
1. Get a free Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys)
2. Enter the API key in the "Settings" section of the app sidebar
3. The app will use this key to generate detailed explanations of detected packets

## 📢 Notes
- The Random Forest model is trained with 10 estimators and max depth of 10 for fast training.
- The dataset file is large (>70MB). For best results, use a machine with sufficient RAM.
- A free Groq API key is required to use the AI explanation feature.
- The app uses Llama 3.3-70B model via Groq for cybersecurity analysis.

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
This project is licensed under the MIT License.

## 🌐 Links
- [GitHub Repo](https://github.com/KRAZATEC/AI_NIDS_Project)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

Made with ❤️ by KRAZATEC
