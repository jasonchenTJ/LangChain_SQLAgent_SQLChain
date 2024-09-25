from langchain.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
import openai
import os
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# 设置你的OpenAI API密钥
api_key="*******"  # 替换为你的OpenAI API密钥
openai.api_key = api_key
# 设置代理环境变量
proxy = "http://127.0.0.1:1087"  # 替换为你的代理地址（需要梯子连接外网）
os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy

llm = OpenAI(temperature=0, verbose=True, openai_api_key=api_key)

username = "postgres"
password = "***"
host = "***"  # 替换为你的PG数据库
port = "5432"
mydatabase = "****"

pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
db = SQLDatabase.from_uri(pg_uri)
print(db._engine.dialect.name)

PROMPT = """ 
Given an input question, first create a syntactically correct postgresql query to run,  
then look at the results of the query and return the answer.  
The question: {question}
"""

question = "please tell me jason's salary from table employee "
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    #db=db,
    verbose=True,
    top_k=1,
    prefix="please connect to db, and run the postgresql, then look at the results of the query and return the answer. if no answers, please return no results and don't retry"
)

resp = agent_executor.invoke(
  question
)