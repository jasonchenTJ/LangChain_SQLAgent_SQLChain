from transformers import AutoModelForCausalLM, AutoTokenizer
#from langchain_core.language_models.llms import LLM
from langchain.llms.base import LLM
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_experimental.sql import SQLDatabaseSequentialChain

class MyCustomLLM2(LLM):
    max_token: int = 4096
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history = []    
    def __init__(self):
        super().__init__()

    def load_model(self,model_name):    
        self.model= AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _call(self, prompt: str, **kwargs):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=4096,pad_token_id=self.tokenizer.eos_token_id,eos_token_id=self.tokenizer.eos_token_id )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt):]
    
    @property
    def _llm_type(self) -> str:
        return "custom-causal-lm"







username = "postgres" 
password = "*********" 
host = "************" 
port = "5432"
mydatabase = "db_hr"

pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
db = SQLDatabase.from_uri(pg_uri)
print(db)

PROMPT = """ 
Given an input question, first create a syntactically correct postgresql query to run,  
then look at the results of the query and return the answer.  
The question: {question}
"""
llm = MyCustomLLM2()
llm.load_model("/root/autodl-tmp/LLM/Llama3-Chinese/Llama3-Chinese")

#db_chain = SQLDatabaseSequentialChain.from_llm(llm=model, db=db, verbose=True, use_query_checker=True,
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True,return_intermediate_steps=True, top_k=1)  
question = "please help query the salary info for user jason?" 
# use db_chain.run(question) instead if you don't have a prompt
result = db_chain(PROMPT.format(question=question)) 
print(result)