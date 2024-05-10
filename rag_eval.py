import os
from datasets.arrow_dataset import Dataset
import pandas as pd
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from datetime import datetime


os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fhsajkdfhsdjahk.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "78690d521a3f48d2a866f5488a05668e"

# questions_input.xlsx laden
input_file = "questions_input.xlsx"
df = pd.read_excel(input_file)

# Spalten überprüfen
# required_columns = ["question", "answer", "ground_truth"]
# if not all(col in df.columns for col in required_columns):
#     raise ValueError(f"Die Eingabedatei muss die Spalten {', '.join(required_columns)} enthalten.")

data_dict = {
        'question': df['question'].tolist(),
        'answer': df['answer'].tolist(),
        'contexts': [[item] for item in df["contexts"].tolist()],
        'ground_truth': df["ground_truth"].tolist()
    }
    
ds = Dataset.from_dict(data_dict)

from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
)

azure_configs = {
    "base_url": "https://fhsajkdfhsdjahk.openai.azure.com/",
    "model_deployment": "chat",
    "model_name": "gpt-35-turbo",
    "embedding_deployment": "embedding",
    "embedding_name": "text-embedding-ada-002",  # most likely
}

azure_model = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["model_deployment"],
    model=azure_configs["model_name"],
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    validate_base_url=False,
)

print("-------------------------")
print(os.getenv("AZURE_OPENAI_API_KEY"))
print(azure_configs["base_url"])
print(azure_configs["model_deployment"])
print(azure_configs["model_name"])
print(azure_configs["embedding_deployment"])
print(azure_configs["embedding_name"])
print("-------------------------")

ragas_azure_model = LangchainLLMWrapper(azure_model)

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-12-01-preview",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["embedding_deployment"],
    model=azure_configs["embedding_name"],
)

# Metrics to use:
metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

# Update metric <answer_relevancy> to Azure:
answer_relevancy.llm = ragas_azure_model
answer_relevancy.embeddings = azure_embeddings

# Update all metrics to ragas Azure model:
for m in metrics:
    m.__setattr__("llm", ragas_azure_model)

result = evaluate(
    ds, metrics=metrics
)

result
print("-------------------------")
print(result)
print("-------------------------")

# Ergebnisse in ragas_result.xlsx speichern
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_file = f"results_{timestamp}.xlsx"
result_df = pd.DataFrame(result, index=[0])
result_df.to_excel(output_file, index=False)

print(f"Die Ergebnisse wurden in {output_file} gespeichert.")