# 📄 Document Compliance Agent

A multi-agent AI application that takes in documents, processes them, and checks for compliance against user-defined rules. Made using LangGraph for the agents and Streamlit for the UI – Made by **Syed Haroon Shoaib**

---

## 🚀 Use this link to open the app

👉 **I have deployed my Streamlit app using Render:**

🔗 https://document-compliance-ai-agent.onrender.com
*(No setup required – just upload documents and start checking compliance.)*

---

## ⚙️ Run locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/document-compliance-agent.git
cd document-compliance-agent
```

### 2. Create and Activate Virtual Environment

#### 🖥 macOS / Linux:

```bash
python3 -m venv dcaa-venv
source dcaa-venv/bin/activate
```

#### 🪟 Windows (CMD):

```cmd
python -m venv dcaa-venv
dcaa-venv\Scripts\activate
```

#### 🪟 Windows (PowerShell):

```powershell
python -m venv dcaa-venv
.\dcaa-venv\Scripts\Activate.ps1
```

### 3. Create a `.env` file 

Create a file named `.env` and add the following variables:

```env
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT= "your_azure_endpoint_here"
AZURE_DOCUMENT_INTELLIGENCE_KEY= "your_azure_key_here"
OPENAI_API_KEY= "your_openai_key_here"
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the App

```bash
streamlit run streamlit_app.py
```

---

## 🧠 Architecture

### Application Pipeline

#### Document Processing
1. User uploads files.
2. Text is extracted using `pdfplumber` and `Azure Document Intelligence OCR`.
3. Text is passed to **DocumentProcessingAgent** to clean and store in ChromaDB.

#### Compliance Checking
1. User inputs compliance rules.
2. Rules go to **ComplianceCheckAgent**, which retrieves relevant docs and checks compliance.
3. A final report is generated and displayed.

### Agents

The program uses 2 LangGraph agents – the DocumentProcessingAgent and the ComplianceCheckAgent.

#### 1. DocumentProcessingAgent
- Takes the extracted text, cleans it, and stores it.
- **Node**: `document_processing_agent` (powered by OpenAI's GPT-4.1)
- **Tools**:
  - `text_cleaning_tool`: Uses LLM to clean raw data.
  - `store_in_vectordb_tool`: Stores cleaned text in ChromaDB.

#### 2. ComplianceCheckAgent
- Takes compliance rules, retrieves relevant docs from the DB, and prepares a report.
- **Node**: `compliance_checking_agent` (also powered by GPT-4.1)
- **Tools**:
  - `find_relevant_docs_tool`: Regex-based tool to extract document names from rules.
  - `retrieve_docs_from_vectordb_tool`: Fetches document text from ChromaDB.

---

## 📘 Description

The Document Compliance Agent is designed to:
- Process PDF and image documents using AI-powered OCR.
- Store extracted content in a vector database.
- Check documents against user-defined compliance rules.
- Provide a clear, consolidated compliance report.

The application supports multiple file uploads, AI-powered processing, and an interactive Streamlit interface for compliance verification and reporting.

## 🧩 Assumptions, Limitations & Future Work

**Assumptions:**

* Users will upload high-quality PDFs or images with recognizable text.
* Compliance rules follow a structured, parseable format (e.g., mention filenames or doc types explicitly).

**Limitations:**

* The app currently supports English documents only.
* Accuracy of document retrieval and compliance matching may vary with rule complexity.
* Rule interpretation is best-effort and not legally exhaustive.

**Future Improvements:**

* Add support for multi-language OCR and compliance rules.
* Introduce feedback loop for human-in-the-loop corrections.
* Improve UI for document browsing and rule-building.

---
