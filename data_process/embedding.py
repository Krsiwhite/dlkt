from openai import OpenAI
import json
import os
import tqdm
client = OpenAI(
    api_key = 'sk-ZCcrX9JD8AfdGohshzZKgPgWV9guGZVJfgAOFy4go1Ctpl7Z',
    base_url = "https://api.chatanywhere.tech/v1"
)

def get_short_embedding(text, model="text-embedding-3-large", dimensions=256):
    text = text.replace("\\n", " ")
    return client.embeddings.create(input=[text], model=model, dimensions=dimensions).data[0].embedding


def getEmbedding(name, Config):
        file_path = os.path.join(Config.dataset.dataPath, name, "used.json")

        with open(file_path, 'r', encoding='utf-8') as f:
            mateTextData = json.load(f)
        for i in tqdm.tqdm(range(len(mateTextData))):
            mateTextData[i]["SubjectEmbedding"] = get_short_embedding(mateTextData[i]["SubjectName"])

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(mateTextData, f, ensure_ascii=False, indent=4)

