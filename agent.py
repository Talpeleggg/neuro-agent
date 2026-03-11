import os
import google.auth
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

def get_credentials():
    """Helper to handle Enterprise Auth Fallback"""
    load_dotenv()
    try:
        credentials, project = google.auth.default()
        return None # Uses ADC
    except google.auth.exceptions.DefaultCredentialsError:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("No K8s Identity and no GOOGLE_API_KEY found in .env")
        return api_key

def get_neural_agent(df, output_dir):
    """The Primary Analysis Agent for chatting and plotting."""
    api_key = get_credentials()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)
    
    instructions = f"""
        You are an elite Computational Neuroscientist and Lead Data Engineer. 
        You specialize in analyzing BCI signals and EEG data.
        
        VISUALIZATION RULES:
        1. Generate production-quality graphs using 'matplotlib' or 'seaborn'.
        2. ALWAYS save the plot as a '.png' file exactly in this directory: '{output_dir}'.
        3. NEVER use `plt.show()`.
        
        CRITICAL FORMATTING RULE:
        You MUST begin your final response with the exact prefix: "Final Answer: ".
        """
        
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True,
        prefix=instructions, handle_parsing_errors=True
    )
    return agent

def generate_data_quality_report(df):
    """The Data Engineering Agent: Automatically profiles data health upon ingestion."""
    api_key = get_credentials()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)
    
    # We pass the descriptive statistics to the LLM to avoid sending the whole raw dataset
    stats_string = df.describe().to_string()
    missing_data = df.isna().sum().to_string()
    
    prompt = f"""
    You are a strict Data Quality Engineer for a neuroscience lab.
    Review the following statistical summary and missing values report for a newly ingested dataset.
    
    Stats:
    {stats_string}
    
    Missing Values:
    {missing_data}
    
    Write a highly concise, 3-bullet-point Data Quality Report. Call out any extreme outliers, significant missing data, or state if the data looks clean and normalized. Do not write a long essay.
    """
    
    response = llm.invoke(prompt)
    return response.content