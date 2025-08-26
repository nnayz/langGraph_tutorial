from fastapi import FastAPI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="Gemma2-9b-It", api_key=groq_api_key)

# 1. Create prompt template
system_template = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{text}")
])

# 2. Create output parser
parser = StrOutputParser()

# 3. Create chain
chain = prompt_template | model | parser

app = FastAPI(title="LangServe Demo", description="A simple LangServe demo", version="0.0.1")

# Adding chain routes to the app
add_routes(
    app, 
    chain, 
    path="/chain",
    enabled_endpoints=["invoke", "stream"]
)

if __name__ == "__main__":
    import uvicorn
    # For reload support without warnings, start via CLI:
    #   uvicorn langserve.serve:app --host localhost --port 8000 --reload
    uvicorn.run(app, host="localhost", port=8000)
