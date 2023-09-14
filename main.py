import os
import pickle

from google.auth.transport.requests import Request

from google_auth_oauthlib.flow import InstalledAppFlow
from llama_index import GPTVectorStoreIndex, LLMPredictor, ServiceContext, download_loader
from langchain import OpenAI


os.environ['OPENAI_API_KEY'] = 'sk-xxxx'


def authorize_gdocs():
    google_oauth2_scopes = [
        "https://www.googleapis.com/auth/documents.readonly"
    ]
    cred = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", 'rb') as token:
            cred = pickle.load(token)
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", google_oauth2_scopes)
            cred = flow.run_local_server(port=0)
        with open("token.pickle", 'wb') as token:
            pickle.dump(cred, token)


if __name__ == '__main__':

    authorize_gdocs()
    GoogleDocsReader = download_loader('GoogleDocsReader')
    gdoc_ids = ['xxxxxxxxxxx']
    loader = GoogleDocsReader()
    documents = loader.load_data(document_ids=gdoc_ids)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=1024))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper,embed_model=embeddings)
    # index = GPTVectorStoreIndex(documents)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    # 将索引保存到 index.json 文件
    # index.save_to_disk( 'target/index.json' ) 
    index.storage_context.persist(persist_dir="target/index.json")
    # # 从保存的 index.json 文件加载索引 index 
    # = GPTSimpleVectorIndex.load_from_disk( 'index.json' )

    query_engine = index.as_query_engine(service_context=service_context)

    while True:
        prompt = input("输入：")
        response = query_engine.query(prompt)
        print(response)

        # Get the last token usage
        last_token_usage = llm_predictor.last_token_usage
        print(f"last_token_usage={last_token_usage}")
