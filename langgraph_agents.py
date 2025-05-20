import streamlit as st
import os
import re
import traceback
import json 
from typing import TypedDict, List, Dict, Optional, Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage 
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition 
from langgraph.graph.message import add_messages 

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool

class DocumentProcessingState(TypedDict):
    file_name: str
    original_text: str
    cleaned_text: Optional[str]
    storage_status: Optional[str]
    messages: Annotated[List, add_messages]
    error_message: Optional[str]

class ComplianceCheckState(TypedDict):
    rules: List[str]
    identified_doc_names: Optional[List[str]]
    retrieved_documents: Optional[List[Dict[str, str]]]
    compliance_report: Optional[List[Dict[str, any]]]
    messages: Annotated[List, add_messages] 
    error_message: Optional[str]


@st.cache_resource
def get_llm_and_embeddings():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY not found in environment variables.")
        return None, None
    try:
        
        llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.2, api_key=openai_api_key)
        embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-large")
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing OpenAI models: {e}")
        return None, None

@st.cache_resource
def get_chromadb_collection():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    try:
        chroma_data_path = "chroma_data"
        os.makedirs(chroma_data_path, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_data_path)

        if not openai_api_key:
            st.error("OpenAI API Key not available for ChromaDB OpenAIEmbeddingFunction.")
            return None

        collection_name = "cleaned_documents_openai"
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-large"
        )
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        return collection
    except ImportError as e_import:
        st.error(f"Failed to import ChromaDB's OpenAIEmbeddingFunction. Ensure chromadb is up to date. Error: {e_import}")
        st.error(traceback.format_exc())
        return None
    except Exception as e:
        st.error(f"Error initializing ChromaDB or its embedding function: {e}")
        st.error(traceback.format_exc())
        return None


# document processing tools
@tool
def text_cleaning_tool(file_name: str, original_text: str) -> str: 
    """
    Cleans the provided text from a document.
    Corrects OCR errors, removes Arabic text, improves formatting.
    Args:
        file_name (str): The name of the file from which text was extracted.
        original_text (str): The original text to be cleaned.
    Returns:
        str: A JSON string containing 'cleaned_text' (str) and 'error_message' (str, optional).
    """
    llm, _ = get_llm_and_embeddings()
    result = {}
    if not llm:
        result = {"cleaned_text": original_text, "error_message": "LLM not available for cleaning."}
        return json.dumps(result)

    prompt_template = f"""You are an expert text cleaning assistant.
The following text was extracted from a document named '{file_name}'.
Please clean the text by:
1. Correcting obvious OCR errors or typos.
2. DISCARD any text that is in ARABIC.
3. Improving formatting for readability (e.g., sensible paragraph breaks, remove excessive newlines).
4. Do NOT remove any substantive content, even if it looks like a header or footer unless it's clearly gibberish.
5. Return only the cleaned text.

Original Text:
---
{original_text}
---
Cleaned Text:"""
    try:
        messages = [
            SystemMessage(content="You are an expert text cleaning assistant."),
            HumanMessage(content=prompt_template)
        ]
        response = llm.invoke(messages)
        cleaned_text = response.content
        result = {"cleaned_text": cleaned_text, "error_message": None}
    except Exception as e:
        result = {"cleaned_text": original_text, "error_message": f"Cleaning failed: {e}. Using original text."}
    return json.dumps(result) 

@tool
def store_in_vectordb_tool(file_name: str, cleaned_text: str) -> str: 
    """
    Stores the cleaned text of a document into the ChromaDB vector store.
    Args:
        file_name (str): The name of the document.
        cleaned_text (str): The cleaned text content of the document.
    Returns:
        str: A JSON string containing 'storage_status' (str) and 'error_message' (str, optional).
    """
    collection = get_chromadb_collection()
    result = {}
    if not collection:
        result = {"storage_status": "Failed: ChromaDB collection unavailable.", "error_message": "ChromaDB collection unavailable."}
        return json.dumps(result)
    if not cleaned_text:
        result = {"storage_status": "Failed: No cleaned text provided to store.", "error_message": "No cleaned text provided to store."}
        return json.dumps(result)

    try:
        doc_id_normalized = file_name.lower()
        metadata = {
            "filename": file_name,
            "source": "doc_upload"
        }
        collection.upsert(
            documents=[cleaned_text],
            metadatas=[metadata],
            ids=[doc_id_normalized]
        )
        result = {"storage_status": f"Successfully stored/updated '{file_name}' in ChromaDB.", "error_message": None}
    except Exception as e:
        print(f"Error storing '{file_name}' in ChromaDB: {e}")
        result = {"storage_status": f"Failed to store '{file_name}' in ChromaDB: {e}", "error_message": f"DB Error: {str(e)}"}
    return json.dumps(result) 

# compliance checking tools
@tool
def find_relevant_doc_names_tool(rules: List[str]) -> str: 
    """
    Identifies unique document filenames mentioned across a list of compliance rules.
    Args:
        rules (List[str]): A list of compliance rule texts.
    Returns:
        str: A JSON string. Contains 'identified_doc_names' (List[str] of unique, normalized document names)
              and 'error_message' (str, optional).
    """
    all_doc_names_in_rules = []
    for rule_text in rules:
        doc_names_in_rule = re.findall(r'([\w\-\_]+\.(?:pdf|txt|docx|xlsx|png|jpg|jpeg|PDF|TXT|DOCX|XLSX|PNG|JPG|JPEG))', rule_text, re.IGNORECASE)
        all_doc_names_in_rules.extend(doc_names_in_rule)

    if not all_doc_names_in_rules:
        return json.dumps({"identified_doc_names": [], "error_message": "No document names explicitly found in any of the rule texts using regex."})

    normalized_unique_doc_names = sorted(list(set(name.lower() for name in all_doc_names_in_rules)))
    return json.dumps({"identified_doc_names": normalized_unique_doc_names, "error_message": None})

@tool
def retrieve_docs_from_vector_db_tool(doc_names: List[str]) -> str: 
    """
    Retrieves specified documents from the ChromaDB vector store by their normalized names.
    Args:
        doc_names (List[str]): A list of normalized document names to retrieve.
    Returns:
        str: A JSON string. Contains 'retrieved_documents' (List[Dict[str, str]], each with 'name', 'content', 'id_in_db')
              and 'error_message' (str, optional).
    """
    collection = get_chromadb_collection()
    retrieved_docs_list = []
    errors_list = []
    result = {}

    if not collection:
        result = {"retrieved_documents": [], "error_message": "ChromaDB collection not available for retrieval."}
        return json.dumps(result)

    if not doc_names:
        result = {"retrieved_documents": [], "error_message": "No document names were provided to fetch."}
        return json.dumps(result)

    doc_ids_to_fetch_normalized = sorted(list(set(name.lower() for name in doc_names)))

    try:
        results_db = collection.get(ids=doc_ids_to_fetch_normalized, include=["documents", "metadatas"]) 
        
        successfully_retrieved_ids_normalized = set(results_db['ids'])

        for queried_normalized_id in doc_ids_to_fetch_normalized:
            if queried_normalized_id in successfully_retrieved_ids_normalized:
                idx = results_db['ids'].index(queried_normalized_id)
                display_name = results_db['metadatas'][idx].get('filename', queried_normalized_id)
                retrieved_docs_list.append({
                    "name": display_name,
                    "content": results_db['documents'][idx],
                    "id_in_db": queried_normalized_id
                })
            else:
                errors_list.append(f"Document '{queried_normalized_id}' (normalized) not found in DB.")
        
        error_message_str = "; ".join(errors_list) if errors_list else None
        result = {"retrieved_documents": retrieved_docs_list, "error_message": error_message_str}
    except Exception as e:
        print(f"Error fetching documents from ChromaDB: {e}\n{traceback.format_exc()}")
        result = {"retrieved_documents": [], "error_message": f"Failed to fetch documents from DB: {e}"}
    return json.dumps(result)



# code for the document processor agent
def document_processor_agent(state: DocumentProcessingState):
    print(f"\n--- Entering document_processor_agent ---")

    llm, _ = get_llm_and_embeddings()

    bound_llm = llm.bind_tools([text_cleaning_tool, store_in_vectordb_tool])
    current_messages = state["messages"]

    print(f"Messages being sent to LLM ({type(current_messages)}):")
    for i, msg in enumerate(current_messages):
        print(f"  Message {i}: role={msg.type}, content='{str(msg.content)[:150]}...', tool_calls={hasattr(msg, 'tool_calls') and msg.tool_calls}")

    try:
        response = bound_llm.invoke(current_messages)
    except Exception as e:
        
        return {"messages": current_messages + [AIMessage(content=f"LLM invocation failed: {e}")], "error_message": (state.get("error_message", "") + f"; LLM invocation failed: {e}").strip("; ")}


    print(f"LLM Response: role={response.type}, content='{str(response.content)[:150]}...', tool_calls={response.tool_calls}")

    
    if response.tool_calls:
        corrected_tool_calls = []
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'text_cleaning_tool':
                print(f"Original args for text_cleaning_tool: {tool_call['args']}") 
                
                tool_call['args']['original_text'] = state['original_text']
                print(f"Corrected args for text_cleaning_tool: {tool_call['args']}") 
            corrected_tool_calls.append(tool_call)
        response.tool_calls = corrected_tool_calls


    new_messages = current_messages + [response]

    if not response.tool_calls:
        
        final_cleaned_text = None
        final_storage_status = None
        accumulated_error = state.get("error_message", "")

        for msg in reversed(new_messages):
            if isinstance(msg, ToolMessage):
                tool_output_content = msg.content
                parsed_tool_output = {}
                try:
                    parsed_tool_output = json.loads(tool_output_content)
                except json.JSONDecodeError:
                    accumulated_error = (accumulated_error + f"; ToolMsg Error: Failed to parse JSON from {msg.name}: {tool_output_content[:100]}...").strip("; ")
                    continue

                if msg.name == "text_cleaning_tool":
                    if "cleaned_text" in parsed_tool_output and final_cleaned_text is None:
                        final_cleaned_text = parsed_tool_output["cleaned_text"]
                        print(f"Extracted cleaned_text (first 200): {final_cleaned_text[:200]}") 
                    if parsed_tool_output.get("error_message"):
                        accumulated_error = (accumulated_error + f"; CleanTool: {parsed_tool_output['error_message']}").strip("; ")
                
                elif msg.name == "store_in_vectordb_tool":
                    if "storage_status" in parsed_tool_output and final_storage_status is None:
                        final_storage_status = parsed_tool_output["storage_status"]
                    if parsed_tool_output.get("error_message"):
                        accumulated_error = (accumulated_error + f"; StoreTool: {parsed_tool_output['error_message']}").strip("; ")
        
        if final_cleaned_text is None and state.get("original_text"): 
            final_cleaned_text = state["original_text"]
            if final_storage_status is None or "No cleaned text" in (final_storage_status or ""):
                 accumulated_error = (accumulated_error + "; Cleaned text not populated by tool, using original. Storage might be affected.").strip("; ")
        
        print(f"LLM decided no more tool calls. Finalizing state.")
        return {
            "messages": new_messages,
            "cleaned_text": final_cleaned_text,
            "storage_status": final_storage_status,
            "error_message": accumulated_error if accumulated_error else None
        }
    
    print(f"LLM made tool calls (possibly corrected). Passing back to graph for tool execution.")
    return {"messages": new_messages, "error_message": state.get("error_message")}


      
@st.cache_resource
def DocumentProcessingAgent():
    workflow = StateGraph(DocumentProcessingState)
    tools = [text_cleaning_tool, store_in_vectordb_tool]
    
    workflow.add_node("document_processor_agent", document_processor_agent)
    tool_node = ToolNode(tools) 
    workflow.add_node("tools_executor", tool_node)

    workflow.set_entry_point("document_processor_agent")
    
    workflow.add_conditional_edges(
        "document_processor_agent", 
        tools_condition,
        { 
            "tools": "tools_executor", 
            END: END                   
        }
    )
    workflow.add_edge("tools_executor", "document_processor_agent") 

    try:
        app = workflow.compile()
        return app
    except Exception as e:
        st.error(f"Error compiling document processing graph: {e}")
        st.error(traceback.format_exc()) 
        return None

    

# code for the compliance checking agent
def compliance_checking_agent(state: ComplianceCheckState):
    llm, _ = get_llm_and_embeddings()
    if not llm:
        error_msg = (state.get("error_message", "") + "; LLM not available for compliance check.").strip("; ")
        return {"messages": state["messages"] + [AIMessage(content="LLM not available.")], "error_message": error_msg}

    bound_llm = llm.bind_tools([find_relevant_doc_names_tool, retrieve_docs_from_vector_db_tool])
    response = bound_llm.invoke(state["messages"])
    new_messages = state["messages"] + [response]

    if not response.tool_calls: 
        final_report = None
        accumulated_error = state.get("error_message", "")
        try:
            
            content_str = response.content
            parsed_report = json.loads(content_str)
            if isinstance(parsed_report, list):
                final_report = parsed_report
            else:
                final_report = [{"rule": "Overall Report", "status": "Needs Review", "justification": f"LLM report not in expected list format. Content: {content_str}", "error_details": "Report format error"}]
                accumulated_error = (accumulated_error + "; LLM report format error").strip("; ")
        except json.JSONDecodeError:
            final_report = []
            for rule_text in state.get("rules", []):
                 final_report.append({
                     "rule": rule_text, "status": "Needs Manual Review",
                     "justification": f"LLM final response was not valid JSON: {response.content}",
                     "error_details": "LLM did not provide a structured JSON report."})
            accumulated_error = (accumulated_error + "; LLM final response not valid JSON.").strip("; ")
        except Exception as e:
            final_report = [{"rule": "Report Generation Error", "status": "Error", "justification": f"Failed to parse LLM final response: {e}", "error_details": str(e)}]
            accumulated_error = (accumulated_error + f"; Report parsing error: {e}").strip("; ")

       
        final_identified_docs = state.get("identified_doc_names")
        final_retrieved_docs = state.get("retrieved_documents")

        for msg in reversed(new_messages):
            if isinstance(msg, ToolMessage):
                tool_content = msg.content
                if isinstance(msg.content, str): 
                    try: tool_content = json.loads(msg.content)
                    except json.JSONDecodeError: pass
                
                if isinstance(tool_content, dict):
                    if msg.name == "find_relevant_doc_names_tool":
                        if "identified_doc_names" in tool_content and not final_identified_docs: 
                            final_identified_docs = tool_content["identified_doc_names"]
                        if tool_content.get("error_message"):
                             accumulated_error = (accumulated_error + f"; FindDocsTool: {tool_content['error_message']}").strip("; ")

                elif msg.name == "retrieve_docs_from_vector_db_tool":
                    if "retrieved_documents" in tool_content and not final_retrieved_docs:
                        final_retrieved_docs = tool_content["retrieved_documents"]
                        print("\n--- Content Retrieved by retrieve_docs_from_vector_db_tool ---")
                        for doc_idx, doc_content in enumerate(final_retrieved_docs):
                            print(f"Document {doc_idx + 1}: {doc_content.get('name')}")
                            print(f"  Content (first 500 chars): {str(doc_content.get('content'))[:500]}")

                            if "NET WEIGHT" in str(doc_content.get('content', '')).upper():
                                print(f"  DEBUG: 'NET WEIGHT' FOUND in retrieved content for {doc_content.get('name')}")
                            else:
                                print(f"  DEBUG: 'NET WEIGHT' NOT FOUND in retrieved content for {doc_content.get('name')}")
                    if tool_content.get("error_message"):
                        accumulated_error = (accumulated_error + f"; RetrieveDocsTool: {tool_content['error_message']}").strip("; ")

        return {
            "messages": new_messages,
            "compliance_report": final_report,
            "identified_doc_names": final_identified_docs,
            "retrieved_documents": final_retrieved_docs,
            "error_message": accumulated_error if accumulated_error else None
        }
    return {"messages": new_messages, "error_message": state.get("error_message")}

@st.cache_resource
def ComplianceCheckAgent():
    workflow = StateGraph(ComplianceCheckState)
    tools = [find_relevant_doc_names_tool, retrieve_docs_from_vector_db_tool]

    workflow.add_node("compliance_checking_agent", compliance_checking_agent)
    workflow.add_node("tools_executor", ToolNode(tools))

    workflow.set_entry_point("compliance_checking_agent")
    workflow.add_conditional_edges(
        "compliance_checking_agent",
        tools_condition,
        {"tools": "tools_executor", END: END}
    )
    workflow.add_edge("tools_executor", "compliance_checking_agent")

    try:

        app = workflow.compile()
        return app
    except Exception as e:
        st.error(f"Error compiling compliance check graph: {e}")
        st.error(traceback.format_exc())
        return None
