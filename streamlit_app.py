import streamlit as st
import os
import re
import tempfile
import traceback
from typing import TypedDict, List, Dict, Optional
import pandas as pd
import json 

import mimetypes
from dotenv import load_dotenv

from fpdf import FPDF

from text_extraction_functions import get_azure_di_client, extract_text
from langgraph_agents import (
    get_llm_and_embeddings,
    get_chromadb_collection,
    DocumentProcessingAgent,
    ComplianceCheckAgent,
    DocumentProcessingState, 
    ComplianceCheckState
)
from langchain_core.messages import HumanMessage 

load_dotenv()

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# this is for converting the agent's output into a pdf
def generate_pdf_report(dataframe: pd.DataFrame) -> bytes:
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Compliance Check Report", 0, 1, 'C')
    pdf.ln(5)

    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(220, 220, 220)

    num_cols = len(dataframe.columns)
    available_width = pdf.w - 2 * pdf.l_margin

    
    relative_widths = [0.30, 0.1, 0.35, 0.15, 0.1]
    if num_cols == len(relative_widths):
         col_widths = [available_width * w for w in relative_widths]
    else:
        
        st.warning(f"PDF column width definition mismatch. Expected {len(relative_widths)} columns, got {num_cols}. Using equal widths.")
        col_widths = [available_width / num_cols] * num_cols


    header_line_height = 7
    for i, col_name in enumerate(dataframe.columns):
        pdf.cell(col_widths[i], header_line_height, col_name, border=1, align='C', fill=True, ln=0)
    pdf.ln(header_line_height)

    pdf.set_font("Arial", "", 8)
    data_line_height = 6

    data_for_table = []
    for _, row_series in dataframe.iterrows():
        data_for_table.append([str(item) for item in row_series])

    
    with pdf.table(
            rows=data_for_table,
            col_widths=tuple(col_widths),
            text_align="LEFT",
            line_height=data_line_height * 1.25, 
            borders_layout="ALL",
            first_row_as_headings=False 
            ) as table:
        pass

    return pdf.output(dest='B')


#streamlit code for the ui
def streamlit_app_main():
    st.set_page_config(layout="wide", page_title="Document Compliance Agent")
    st.title("📑 Document Compliance Agent")
    st.markdown("##### Made by Syed Haroon Shoaib")

    azure_di_client = get_azure_di_client()
    llm, embeddings = get_llm_and_embeddings()
    doc_processing_graph = DocumentProcessingAgent()
    chroma_collection = get_chromadb_collection()
    full_compliance_graph = ComplianceCheckAgent()

    services_ok = True

    if not OPENAI_API_KEY:
        st.sidebar.error("OpenAI API Key missing from .env!")
        services_ok = False
    if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or not AZURE_DOCUMENT_INTELLIGENCE_KEY:
        st.sidebar.warning("Azure DI Endpoint/Key missing from .env. PDF/Image OCR will fail.")

    if not azure_di_client and (AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY) :
        st.sidebar.error("Azure DI client failed to initialize despite keys being present.")
        services_ok = False
    if not llm or not embeddings:
        st.sidebar.error("OpenAI models (LLM/Embeddings) failed to initialize.")
        services_ok = False
    if not doc_processing_graph:
        st.sidebar.error("Doc processing graph failed to compile.")
        services_ok = False
    if not chroma_collection:
        st.sidebar.error("ChromaDB collection failed to initialize.")
        services_ok = False
    if not full_compliance_graph:
        st.sidebar.error("Full compliance graph failed to compile.")
        services_ok = False

    with st.sidebar:
        st.markdown("### REMEMBER:")
        st.markdown("1. Upload and process documents in the **📄 Document Processing** Tab.")
        st.markdown("2. Then, go to the **⚖️ Compliance Checking** Tab to enter rules and run checks.")
        st.markdown("---")
        st.markdown("### NOTE:")
        st.markdown("When entering compliance rules, ensure document names are mentioned with the file extentions (e.g., customs.pdf) at least for the first mention of them")

    tab1, tab2 = st.tabs(["📄 Document Processing", "⚖️ Compliance Checking"])

    with tab1:
        st.header("Document Processing")
        st.markdown("Upload documents to extract text, clean it, and store it in the vector database using an AI agent.")

        if 'extraction_results' not in st.session_state: st.session_state.extraction_results = {}
        if 'processed_file_ids_ingestion' not in st.session_state: st.session_state.processed_file_ids_ingestion = set()

        uploaded_files_ingestion = st.file_uploader(
            "Choose documents",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
            accept_multiple_files=True,
            key="ingestion_uploader"
        )

        if uploaded_files_ingestion:
            files_to_process_ingestion = [f for f in uploaded_files_ingestion if f.file_id not in st.session_state.get('processed_file_ids_ingestion', set())]

            if files_to_process_ingestion:
                st.markdown(f"**New files ready for processing ({len(files_to_process_ingestion)}):**")
                for f_obj in files_to_process_ingestion: st.markdown(f"- `{f_obj.name}`")

                if st.button("🚀 Process Uploaded Documents", type="primary", disabled=not services_ok):
                    with st.spinner("Processing documents... This may take a while."):
                        for uploaded_file_obj in files_to_process_ingestion:
                            file_info = {
                                'id': uploaded_file_obj.file_id,
                                'name': uploaded_file_obj.name,
                                'data': uploaded_file_obj.getvalue()
                            }

                            st.markdown(f"--- \nProcessing: `{file_info['name']}`")
                            result_entry = {
                                "name": file_info['name'], "original_text": None, "cleaned_text": None,
                                "storage_status": "Not processed", "error": None, "traceback_str": None
                            }
                            tmp_file_path = None
                            try:
                                file_extension = os.path.splitext(file_info['name'])[1]
                                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                                    tmp_file.write(file_info['data'])
                                    tmp_file_path = tmp_file.name

                                st.info(f"Extracting text from: {file_info['name']} (type: {file_extension})")
                                original_text_content = extract_text(tmp_file_path, azure_di_client)
                                result_entry["original_text"] = original_text_content

                                if not original_text_content or not original_text_content.strip():
                                    warning_msg = f"No text extracted from {file_info['name']}."
                                    if file_extension == '.txt':
                                        warning_msg = f"The text file {file_info['name']} appears to be empty or contains only whitespace."
                                    st.warning(warning_msg)
                                    result_entry["error"] = (result_entry["error"] or "") + warning_msg
                                else:
                                    st.success(f"Text extraction successful for {file_info['name']}.")
                                    st.info(f"Invoking AI agent to clean text and store in vector db for {file_info['name']}...")
                                    if doc_processing_graph:
                                        
                                        initial_doc_processing_input = DocumentProcessingState(
                                            file_name=file_info['name'],
                                            original_text=original_text_content, 
                                            cleaned_text=None,
                                            storage_status=None,
                                            messages=[ 
                                                HumanMessage(
                                                    content=(
                                                        f"You are a document processing assistant for the document named '{file_info['name']}'.\n"
                                                        f"The full original text of this document has been extracted and is provided in the 'original_text' field of the current state.\n" # Be more explicit
                                                        f"Your tasks are:\n"
                                                        f"1. Call the 'text_cleaning_tool'. For its 'file_name' argument, use '{file_info['name']}'. For its 'original_text' argument, you MUST use the value from the 'original_text' field of the current state.\n"
                                                        f"2. After the 'text_cleaning_tool' provides the cleaned text, call the 'store_in_vectordb_tool'. For its 'file_name' argument, use '{file_info['name']}'. For its 'cleaned_text' argument, use the cleaned text obtained from the 'text_cleaning_tool'.\n"
                                                        f"Follow these steps sequentially. Provide a final confirmation after successful storage."
                                                    )
                                                )
                                            ],
                                            error_message=None
                                        )
                                       
                                        graph_output: DocumentProcessingState = doc_processing_graph.invoke(initial_doc_processing_input)
                                        
                                        result_entry["cleaned_text"] = graph_output.get("cleaned_text", original_text_content)
                                        result_entry["storage_status"] = graph_output.get("storage_status", "Processing error by agent")
                                        
                                        agent_error = graph_output.get("error_message")
                                        if agent_error:
                                            result_entry["error"] = (result_entry["error"] + "; " if result_entry["error"] else "") + f"Agent Error: {agent_error}"
                                            st.warning(f"Issues during AI agent processing for {file_info['name']}: {agent_error}")
                                        else:
                                            st.success(f"AI Agent successfully processed and stored {file_info['name']}.")
                                    else:
                                        err_msg = "Document processing agent (LangGraph) not available."
                                        result_entry["error"] = (result_entry["error"] + "; " if result_entry["error"] else "") + err_msg
                                        result_entry["cleaned_text"] = original_text_content
                                        st.warning(f"{err_msg} Skipping AI agent processing for {file_info['name']}.")

                            except Exception as e_main_loop:
                                print(f"Critical error processing {file_info['name']} in main loop: {e_main_loop}")
                                tb_str = traceback.format_exc()
                                result_entry["error"] = (result_entry["error"] or "") + f" Critical error: {str(e_main_loop)}"
                                result_entry["traceback_str"] = tb_str
                                st.error(f"Critical error with {file_info['name']}: {e_main_loop}")
                                st.code(tb_str)
                            finally:
                                if tmp_file_path and os.path.exists(tmp_file_path):
                                    os.remove(tmp_file_path)
                                st.session_state.extraction_results[file_info['id']] = result_entry
                                st.session_state.processed_file_ids_ingestion.add(file_info['id'])
                        st.success("✅ Document processing batch complete!");
            elif uploaded_files_ingestion:
                 st.info("All currently uploaded files have been processed in this session. Upload new files or clear session to reprocess.")

        if st.session_state.extraction_results:
            st.markdown("---"); st.subheader("📜 Document Processing Summaries:")
            for file_id, res_data in st.session_state.extraction_results.items():
                expander_title = f"Summary for: `{res_data['name']}`"
                if res_data.get('error'):
                    expander_title += " - ⚠️ Issues"
                elif res_data.get('storage_status', '').startswith("Successfully"):
                    expander_title += " - ✅ Processed & Stored"
                else:
                    expander_title += f" - Status: {res_data.get('storage_status', 'N/A')}"

                with st.expander(expander_title, expanded=False):
                    if res_data.get('error'): st.error(f"Error Summary: {res_data['error'].strip()}")
                    if res_data.get('traceback_str'): st.code(res_data['traceback_str'], language='python')

                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area("Original Text Snippet", value=(res_data.get('original_text', "N/A") or "")[:1000]+"...", height=150, key=f"orig_ingest_{file_id}", disabled=True)
                    with col2:
                        st.text_area("Cleaned Text Snippet (from Agent)", value=(res_data.get('cleaned_text', "N/A") or "")[:1000]+"...", height=150, key=f"clean_ingest_{file_id}", disabled=True)
                    st.caption(f"Agent DB Storage Status: {res_data.get('storage_status', 'N/A')}")

    with tab2:
        st.header("Compliance Checking")
        st.markdown("Enter compliance rules (one per line). The AI agent will retrieve relevant documents and check compliance for all rules, providing a consolidated report. Remeber to use the file extentions (eg. .pdf) with the file name, at least for the first mention of the file")

        if 'compliance_check_results_list' not in st.session_state:
            st.session_state.compliance_check_results_list = []
        if 'overall_compliance_error' not in st.session_state:
            st.session_state.overall_compliance_error = None
        if 'retrieved_docs_for_compliance' not in st.session_state:
            st.session_state.retrieved_docs_for_compliance = []


        rules_input = st.text_area(
            "Enter Compliance Rules (one per line)",
            height=150,
            key="compliance_rules_input",
            placeholder="E.g., 1. customs.pdf should have the same net weight as commercial_invoice.pdf\n2. Customs.pdf should have the net weight 1522 kg"
        )

        if st.button("🔍 Check Compliance for All Rules", type="primary", disabled=not services_ok or not rules_input.strip()):
            st.session_state.compliance_check_results_list = []
            st.session_state.overall_compliance_error = None
            st.session_state.retrieved_docs_for_compliance = []

            rules_list = [rule.strip() for rule in rules_input.splitlines() if rule.strip()]

            if not rules_list:
                st.warning("Please enter some compliance rules.")
            else:
                st.markdown(f"--- \n**Processing {len(rules_list)} rule(s) with the AI Compliance Agent...**")
                
                rules_for_prompt = "\n".join([f"- {r}" for r in rules_list])
                initial_prompt_content = (
                    f"You are a meticulous compliance checking AI. You are given a list of compliance rules.\n"
                    f"The rules are:\n{rules_for_prompt}\n\n"
                    f"Your overall process is as follows:\n"
                    f"1. First, use the 'find_relevant_doc_names_tool' with ALL the rules provided above to identify all unique document filenames mentioned across them.\n"
                    f"2. Next, use the 'retrieve_docs_from_vector_db_tool' to fetch the content of ALL these identified documents from the vector database.\n"
                    f"3. After retrieving the documents, for EACH rule in the original list, you must assess its compliance based on the content of the relevant retrieved documents. If a document specifically mentioned in a rule was not found or not retrieved, this should be noted in your assessment for that rule.\n"
                    f"4. Your final answer MUST be a JSON formatted list of objects. Each object in the list represents the compliance check result for one of the original rules and MUST have the following keys:\n"
                    f"   - 'rule': The exact original rule text (string).\n"
                    f"   - 'status': Your compliance assessment for that rule (e.g., 'Compliant', 'Non-Compliant', 'Needs Manual Review', 'Error: Document Not Found') (string).\n"
                    f"   - 'justification': Your  reasoning for the status in as short as possible, referencing specific document parts if possible. Also mention which documents were primarily used for this rule's assessment (string).\n"
                    f"   - 'error_details': Any errors encountered specifically during the processing of this rule or its documents (string, or null if no errors for this specific rule).\n\n"
                    f"Ensure your final output is ONLY the JSON list. Do not add any introductory or concluding text outside the JSON structure."
                    f"If you cannot perform a step (e.g., a tool fails), try to continue with the next steps if possible, or reflect the failure in the 'error_details' or 'justification' for the affected rules in your final JSON report."
                )

                initial_compliance_input = ComplianceCheckState(
                    rules=rules_list,
                    identified_doc_names=None, 
                    retrieved_documents=None, 
                    compliance_report=None,   
                    error_message=None,       
                    messages=[HumanMessage(content=initial_prompt_content)]
                )
                
                graph_output: Optional[ComplianceCheckState] = None

                if full_compliance_graph:
                    with st.spinner(f"AI Agent is processing {len(rules_list)} rules... This might take some time."):
                        try:
                            graph_output = full_compliance_graph.invoke(initial_compliance_input)
                        except Exception as e_graph_invoke:
                            st.error(f"Error invoking AI compliance agent: {e_graph_invoke}")
                            st.session_state.overall_compliance_error = f"Agent invocation failed: {e_graph_invoke}"
                else:
                    st.error("AI compliance agent (LangGraph) is not available.")
                    st.session_state.overall_compliance_error = "Compliance agent unavailable."

                if graph_output:
                    st.session_state.compliance_check_results_list = graph_output.get('compliance_report', [])
                    st.session_state.overall_compliance_error = graph_output.get('error_message')
                    st.session_state.retrieved_docs_for_compliance = graph_output.get('retrieved_documents', [])

                    if st.session_state.overall_compliance_error:
                        st.warning(f"Overall AI Agent Note/Error: {st.session_state.overall_compliance_error}")
                    
                    if not st.session_state.compliance_check_results_list and not st.session_state.overall_compliance_error:
                        st.info("Agent completed but did not return a compliance report. Check agent logs or prompt.")
                    elif st.session_state.compliance_check_results_list:
                         st.success(f"✅ AI Agent processing complete! {len(st.session_state.compliance_check_results_list)} rule(s) assessed.")
                else: 
                    if not st.session_state.overall_compliance_error: 
                        st.session_state.overall_compliance_error = "Critical error: Agent did not return any output."
                    st.error(st.session_state.overall_compliance_error)


        if st.session_state.get('overall_compliance_error'):
            st.error(f"**Overall Agent Processing Error/Note:** {st.session_state.overall_compliance_error}")

        if st.session_state.get('retrieved_docs_for_compliance'):
            st.info(f"**Documents retrieved by the agent for these rules:** "
                    f"{', '.join([doc['name'] for doc in st.session_state.retrieved_docs_for_compliance]) or 'None'}")


        if st.session_state.compliance_check_results_list:
            st.markdown("---"); st.subheader("⚖️ Consolidated Compliance Check Results:")
            
            report_data_for_df = []
            for res_item in st.session_state.compliance_check_results_list:

                if not isinstance(res_item, dict):
                    st.warning(f"Skipping malformed result item: {res_item}")
                    report_data_for_df.append({
                        "Rule": "Malformed result item from agent",
                        "Status": "Error",
                        "Justification": str(res_item),
                        "Documents Considered (see Justification)": "N/A",
                        "Errors/Notes": "Agent returned non-dict item in report list."
                    })
                    continue

                report_data_for_df.append({
                    "Rule": res_item.get('rule', 'N/A'),
                    "Status": res_item.get('status', 'Error'),
                    "Justification": res_item.get('justification', 'N/A'),

                    "Documents Considered (see Justification)": "Per rule justification",
                    "Errors/Notes": res_item.get('error_details', "None")
                })

            if report_data_for_df:
                df_results = pd.DataFrame(report_data_for_df)
                st.dataframe(df_results, use_container_width=True)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv_data = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Report (CSV)", data=csv_data, file_name="compliance_report.csv", mime="text/csv", key="download_csv", use_container_width=True)
                with col_dl2:
                    try:
                        pdf_bytes = generate_pdf_report(df_results)
                        st.download_button(label="📄 Download Report (PDF)", data=bytes(pdf_bytes), file_name="compliance_report.pdf", mime="application/pdf", key="download_pdf", use_container_width=True)
                    except Exception as e_pdf:
                        st.error(f"Failed to generate PDF report: {e_pdf}")
                        st.code(traceback.format_exc())
            else:
                st.info("No valid compliance results to display or download from the agent's report.")

            st.markdown("---")
            st.subheader("Detailed Breakdown per Rule (from Agent's Report):")
            for i, result_item in enumerate(st.session_state.compliance_check_results_list):
                if not isinstance(result_item, dict): 
                    st.warning(f"Skipping malformed result item in detailed breakdown: {result_item}")
                    continue

                rule_text = result_item.get('rule', 'N/A')
                status = result_item.get('status', 'Error')
                exp_title = f"Rule: `{rule_text[:60]}{'...' if len(rule_text) > 60 else ''}` - Status: {status}"
                with st.expander(exp_title, expanded=False):
                    st.markdown(f"**Full Rule:** {rule_text}")
                    st.markdown(f"**Status:** {status}")
                    st.markdown("**Justification & Documents Considered:**")
                    st.markdown(result_item.get('justification', 'N/A'))
                    
                    error_details = result_item.get('error_details')
                    if error_details and error_details.lower() != "none" and error_details.strip() != "":
                        st.error(f"Rule-Specific Error/Note: {error_details}")
        elif not st.session_state.get('overall_compliance_error') and rules_input.strip() and st.session_state.button_states["🔍 Check Compliance for All Rules"]: # Check if button was pressed
             st.info("Compliance check initiated, but no results were populated. The agent might have encountered an issue not caught as an overall error, or returned an empty report.")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        st.error("FATAL: OPENAI_API_KEY environment variable is not set. The application cannot start.")
        st.stop()
    if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or not AZURE_DOCUMENT_INTELLIGENCE_KEY:
        st.warning("Azure Document Intelligence environment variables (AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY) are not set. Text extraction for PDFs and images will likely fail or use limited fallbacks.")

    
    if "button_states" not in st.session_state:
        st.session_state.button_states = {}

    
    def on_button_click(button_key):
        st.session_state.button_states[button_key] = True

    streamlit_app_main()
