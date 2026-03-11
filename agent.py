import os
import google.auth
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

def get_neural_agent(df, output_dir):
    load_dotenv()
    
    api_key = None
    try:
        credentials, project = google.auth.default()
        print("System: Using Kubernetes/GCP Workload Identity.")
    except google.auth.exceptions.DefaultCredentialsError:
        print("System: ADC not found. Falling back to local .env API key.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("No K8s Identity and no GOOGLE_API_KEY found in .env")

    # Pass the api_key explicitly to prevent the 400 error
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)
    
    instructions = f"""
        You are an elite Computational Neuroscientist and Lead Data Engineer. 
        You specialize in analyzing Brain-Computer Interface (BCI) signals and EEG data.
        
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