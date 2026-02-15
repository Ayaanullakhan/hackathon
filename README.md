# Sophiie AI Agents Hackathon 2026

## Participant

**Name:** Ayaanulla Khan\
**University:** Deakin University

------------------------------------------------------------------------

# Site PPE Spotter --- AI Construction & Mining Safety Agent

An AI-powered Personal Protective Equipment (PPE) compliance monitoring
system designed for high-risk environments such as construction sites
and mining operations.

This project was built for the **Sophiie AI Agents Hackathon 2026**.

------------------------------------------------------------------------

## 🚧 The Problem

In construction and mining:

-   PPE compliance is manually monitored\
-   Supervisors cannot observe all workers continuously\
-   Violations often go unnoticed\
-   Human inspection is inconsistent and reactive

------------------------------------------------------------------------

## 💡 Our Solution

An AI safety agent that:

1.  Detects workers and PPE (hardhat, vest)\
2.  Assigns PPE items to specific workers using spatial reasoning\
3.  Calculates compliance percentage per worker\
4.  Determines risk level (LOW / MEDIUM / HIGH)\
5.  Generates an AI-powered safety audit report

------------------------------------------------------------------------

## 🧠 System Architecture

**1️⃣ Vision Layer**\
Custom-trained YOLOv11 model for detecting: - Person\
- Hardhat\
- Vest

**2️⃣ Compliance Engine**\
- Associates PPE to workers using bounding-box logic\
- Evaluates against site safety policy\
- Generates structured compliance output

**3️⃣ AI Report Layer**\
- OpenAI GPT-4o-mini generates readable safety audit summaries

**4️⃣ UI Layer**\
- Streamlit interactive interface

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python\
-   YOLOv11 (Ultralytics)\
-   Streamlit\
-   OpenAI API

------------------------------------------------------------------------

## 📂 Project Structure

    hackathon/
    ├── train.py
    ├── PPE_detector.py
    ├── compliance_rules.py
    ├── openai_agent.py
    ├── app_streamlit.py
    ├── datasets/
    ├── runs/
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🚀 How It Works

Upload Image\
↓\
YOLO detects people and PPE\
↓\
PPE assigned to workers\
↓\
Compliance calculated\
↓\
Risk level determined\
↓\
AI generates safety audit report

------------------------------------------------------------------------

## 🖥 Installation

### 1) Clone

    git clone https://github.com/Ayaanullakhan/hackathon.git
    cd hackathon

### 2) Create Virtual Environment (Windows)

    python -m venv .venv
    . .\.venv\Scripts\Activate.ps1

### 3) Install Dependencies

    pip install -r requirements.txt

### 4) Add Environment Variable (.env)

    OPENAI_API_KEY=your_key_here
    OPENAI_MODEL=gpt-4o-mini

### 5) Run App

    streamlit run app_streamlit.py

------------------------------------------------------------------------

## 📈 Future Improvements

-   Real-time CCTV video monitoring\
-   Additional PPE classes (gloves, boots, goggles)\
-   Pose estimation for better assignment\
-   Dashboard analytics

------------------------------------------------------------------------

## 👷 Impact

PPE saves lives.\
AI ensures it is worn.
