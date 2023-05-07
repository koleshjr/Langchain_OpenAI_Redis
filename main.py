import os
import redis
import time
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import TagField
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from consts import llm_model_openai,sentence_transformer_model,ITEM_KEYWORD_EMBEDDING_FIELD

load_dotenv()
openai_api_key = os.getenv("OPEN_API_KEY")
redis_password = os.getenv("REDIS_PASSWORD")
redis_host = os.getenv("REDIS_HOST")

#connect to redis cloud
redis_conn = redis.Redis(
  host=redis_host,
  port=11967,
  password=redis_password)
#create vectors using sentence transformers
model = SentenceTransformer(sentence_transformer_model)


llm = ChatOpenAI(model_name=llm_model_openai, temperature=0.3, openai_api_key=openai_api_key)
prompt = PromptTemplate(
    input_variables=["product_description"],
    template="Create comma seperated product keywords to perform a query on a amazon dataset for this user input: {product_description}",
)


chain = LLMChain(llm=llm, prompt=prompt)

userinput = input("Hey im a E-commerce Chatbot, how can i help you today? ")
print("User:", userinput)
# Run the chain only specifying the input variable.
keywords = chain.run(userinput)

topK = 3
# Vectorize the query
query_vector = model.encode(keywords).astype(np.float32).tobytes()

# Prepare the query
q = Query(f'*=>[KNN {topK} @{ITEM_KEYWORD_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0, topK).return_fields('vector_score', 'item_name', 'item_id', 'item_keywords').dialect(2)
params_dict = {"vec_param": query_vector}

# Execute the query
results = redis_conn.ft().search(q, query_params=params_dict)

full_result_string = ''
for product in results.docs:
    full_result_string += product.item_name + ' ' + product.item_keywords + ' ' + product.item_id + "\n\n\n"

from langchain.memory import ConversationBufferMemory

template = """You are a chatbot. Be kind, detailed and nice. Present the given queried search result in a nice way as answer to the user input. dont ask questions back! just take the given context

{chat_history}
Human: {user_msg}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_msg"],
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")
llm_chain = LLMChain(
    llm=ChatOpenAI(model_name=llm_model_openai, temperature=0.8, openai_api_key=openai_api_key),
    prompt=prompt,
    verbose=False,
    memory=memory,
)

answer = llm_chain.predict(user_msg=f"{full_result_string} ---\n\n {userinput}")
print("Bot:", answer)
time.sleep(0.5)

while True:
    follow_up = input("Anything else you want to ask about this topic?")
    print("User:", follow_up)
    answer = llm_chain.predict(
        user_msg=follow_up
    )
    print("Bot:", answer)
    time.sleep(0.5)
