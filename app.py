from flask import Flask, render_template, request

app = Flask(__name__)

from openai.embeddings_utils import cosine_similarity
from openai.embeddings_utils import get_embedding
import openai
import pandas as pd
import numpy as np
import pinecone

api_key = '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'

def generateAnswer(question):
    question_vector = get_embedding(question,engine = 'text-embedding-ada-002')

    df = pd.read_csv('/Users/adilqaisar/Documents/Zacks/AAPLEER_embedded.csv')

    embedding_array = np.load('/Users/adilqaisar/Documents/Zacks/compressed_AAPLEER_embedded_array.npz', mmap_mode='r',allow_pickle=True)
    df['embedding_array'] = embedding_array['data'].tolist()
        
        #df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    df["similarities"] = df['embedding_array'].apply(lambda x: cosine_similarity(x, question_vector))

    df = df.sort_values("similarities",ascending = False).head(5)


    context = []

    for i, row in df.iterrows():
        context.append(row['sentences'])


    text = "\n".join(context)

    context = text

    prompt = f"""Answer the following question using only the context below. Answer as if you're explaining to a 12 year old with no knowledge of investing. If you don't know the answer, say I don't know, don't make up an answer.

    Context:
    {context}

    Q: {question}
    A:"""


    AIAnswer = (openai.Completion.create(
        prompt=prompt,
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model="text-davinci-003"
        )["choices"][0]["text"].strip(" \n"))

    return "Bot: "+AIAnswer


openai.api_key= api_key

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        question = request.form['question']
        answer = generateAnswer(question)
        return render_template('chat.html', question=question, answer=answer)
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)