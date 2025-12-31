
# 🚦 AI_NIDS_Project: AI-based Network Intrusion Detection System 🚦

Welcome to the **AI_NIDS_Project**! This project leverages the power of Artificial Intelligence and Machine Learning to detect and classify network intrusions in real-time. Whether you're a cybersecurity enthusiast, data scientist, or network admin, this tool is designed to help you analyze, visualize, and secure your network traffic with ease. 🛡️

## ✨ Features
- 🔍 **Data Preprocessing & Feature Engineering**: Clean and transform raw network data for optimal model performance.
- 🤖 **Machine Learning Models**: Train, evaluate, and deploy models to detect various types of network attacks.
- 📊 **Interactive Visualizations**: Explore your data and model results with beautiful charts powered by Seaborn and Matplotlib.
- 🖥️ **Streamlit Web App**: User-friendly interface for real-time and batch intrusion detection.
- 📈 **Performance Metrics**: Get detailed reports on accuracy, precision, recall, and more.

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
   pip install pandas numpy scikit-learn streamlit seaborn matplotlib
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
- seaborn
- matplotlib

## 💡 How It Works
1. **Load Data**: Import the CSV dataset containing network traffic.
2. **Preprocess**: Clean, normalize, and engineer features.
3. **Train Model**: Use machine learning algorithms to learn attack patterns.
4. **Detect Intrusions**: Predict and classify new network traffic as normal or attack.
5. **Visualize**: Explore results and metrics interactively.

## 📢 Notes
- The dataset file is large (>70MB). For best results, use a machine with sufficient RAM.
- For files over 100MB, consider using [Git LFS](https://git-lfs.github.com/).

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
