

# 📄 **AeroSense RAG — UAV Troubleshooting Assistant (Manuals + Telemetry + Local LLM)**

*A Multi-Modal RAG System for Diagnostics using Engineering Manuals & Sensor Logs*
![Python](https://img.shields.io/badge/Python-3.11-blue)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![Transformers](https://img.shields.io/badge/Embeddings-MiniLM%20%2F%20GTE--base-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20\(TinyLlama%2FQwen\)-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 🚀 **Project Overview**

**AeroSense RAG** is a multi-modal **Retrieval-Augmented Generation (RAG)** system designed to diagnose UAV issues by combining:

* Engineering **manuals** (TXT/PDF)
* **Telemetry logs** (IMU, GPS, ESC, RPM, Voltage)
* **Local LLM inference** (TinyLlama/Qwen via Ollama)

This system retrieves relevant manual sections and telemetry patterns, blends them semantically, and generates troubleshooting insights — **fully offline**, engineered for **aerospace, robotics, and defense-grade environments**.

### 🎯 **Primary Capabilities**

* Parse and chunk UAV engineering manuals
* Convert telemetry logs into RAG-searchable text
* Build a vector database with 100+ manual chunks and 60k+ telemetry entries
* Perform multi-modal semantic search (manual + logs)
* Rank results using weighted retrieval
* Diagnose faults using a local LLM
* Provide real-time insights through a Streamlit dashboard

---

# 🧠 **Key Use Cases**

* ESC overheating during climb
* GPS dropout / HDOP spikes
* IMU vibration anomalies
* Motor desync or RPM drop
* Voltage sag under load
* Propeller imbalance
* Communication drop / failsafe

---

# 🏗️ **Architecture Overview**

```
          ┌───────────────────┐
          │  Engineering      │
          │  Manuals (PDF/TXT)│
          └─────────┬─────────┘
                    │
                    ▼
            Text Extraction + Chunking
                    │
                    ▼
      ┌───────────────────────────────────┐
      │  Embeddings (MiniLM / GTE-base)  │
      └────────────────┬──────────────────┘
                       │
                       ▼
               ChromaDB Vector Store
                       │
                       │ retrieve top-k
                       ▼
            Weighted Multi-Modal Ranking
                       │
                       ▼
              Local LLM (Ollama)
                       │
                       ▼
              Troubleshooting Output
```

---

# 🧰 **Tech Stack**

### **Core**

* Python 3.11
* SentenceTransformers (MiniLM-L6, GTE-base)
* ChromaDB
* Ollama (TinyLlama, Qwen)
* Streamlit

### **Data Ingestion**

* csv
* pdfplumber
* pathlib

### **Evaluation**

* MRR
* Precision@K
* Recall@K

### **Other**

* Pandas / NumPy
* Local vector database (persistent mode)

---

# 📦 **Project Structure**

```
aerosense_rag/
│
├── app/
│   └── streamlit_app.py
│
├── rag_pipeline/
│   ├── config.py
│   ├── data_ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── llm_inference.py
│   └── evaluation.py
│
├── scripts/
│   └── build_index.py
│
├── data/
│   ├── manuals/          # excluded via .gitignore
│   ├── logs/             # excluded via .gitignore
│   └── ground_truth/
│
├── chroma_db/            # excluded via .gitignore
│
├── requirements.txt
└── README.md
```

---

# ⚙️ **Installation**

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/aerosense_rag.git
cd aerosense_rag
```

### 2. Create venv

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install pdfplumber

```bash
pip install pdfplumber
```

---

# 🗃️ **Add Your Data**

Place engineering manuals (PDF/TXT):

```
data/manuals/
    ├── mannual_2.txt
    └── mannual_1.txt
```

Place telemetry logs (CSV):

```
data/logs/
    flight01_normal.csv
    flight02_overheat.csv
    flight03_gps_drop.csv
    imu_100hz_50000rows.csv
    biglog_10000rows.csv
```

---

# 🧱 **Build the Vector Database**

Run:

```bash
python -m scripts.build_index
```

You should see output like:

```
✔ Manual chunks: 108
✔ Telemetry chunks: 60900
[SUCCESS] Collection 'manual_chunks' built with 108 items.
[SUCCESS] Collection 'telemetry_records' built with 60900 items.
```

---

# 🖥️ **Run the Streamlit App**

```bash
streamlit run app/streamlit_app.py
```

This opens the interactive dashboard:

* enter UAV fault description
* see retrieved manual + telemetry context
* get troubleshooting insights

---

# 🔍 **Sample Queries**

Try these inside the UI:

```
ESC overheating during high-altitude climb
IMU vibration spikes at 40–60s mark
GPS dropout after aggressive yaw maneuver
Motor desync causing RPM imbalance
Voltage sag under high throttle load
```

The system will retrieve multi-modal context and generate a synthesized explanation using the local LLM.

---

# 📊 **Retrieval Evaluation**

Use:

```bash
python -m rag_pipeline.evaluation
```

Reports:

* **Precision@5**
* **MRR**
* Candidate ranking visualization

---

# 📸UI_Screenshots 

![UI](/images/1.png)
![UI_Retr](/images/2.pngL)






---

# 🛠️ **Future Enhancements**

* PID anomaly detection
* Flight-envelope visualizer
* Vibration spectrum analysis (FFT)
* Re-ranking using cross-encoders
* Model distillation for faster edge inference
* Integration with ROS2 or MAVLink parsing

---

# 📜 **License**

MIT License – free for personal & commercial use.





