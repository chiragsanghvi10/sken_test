#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, request
app = Flask(__name__)
import pandas as pd
from rasa.nlu.training_data import load_data
from rasa.nlu.model import Trainer
from rasa.nlu import config
from rasa.nlu.model import Interpreter
import tensorflow.keras as keras
from flask import jsonify
from flask import Response
import os


# In[ ]:


print(os.getcwd())


# In[ ]:


# input_excel_file with column names [speaker & text]
#Speaker containing Agent & Customer

@app.route("/agent_customer_seq",methods=["POST","GET"])

def agent_customer(input_excel_file, output_dir): 
    df = pd.read_excel(input_excel_file)
    df.text= df.text.astype(str)
    df['a_bin'] = 0
    df['b_bin'] = 0
    df.a_bin = df.speaker.apply(lambda x: 0 if x=='Agent' else 1)
    df.b_bin = df.speaker.apply(lambda x: 0 if x=='Customer' else 1)
    df['a_bin_cumsum'] = df.a_bin.cumsum()
    df['b_bin_cumsum'] = df.b_bin.cumsum()
    df = df.drop(['a_bin','b_bin'],axis=1)
    df['a_bin'] = df.speaker.apply(lambda x: 1 if x=='Agent' else 0)
    df['b_bin'] = df.speaker.apply(lambda x: 1 if x=='Customer' else 0)
    df['a_con'] = df.a_bin_cumsum*df.a_bin 
    df['b_con'] = df.b_bin_cumsum*df.b_bin 
    df.drop(['a_bin_cumsum','b_bin_cumsum','a_bin','b_bin'],axis=1,inplace=True)
    df['identifier'] = df.a_con + df.b_con
    df['name_idnet'] = df.speaker+"_"+df.identifier.astype(str)
    df.drop(['a_con','b_con'],axis=1,inplace=True)
    df1 = df[['name_idnet','text']].groupby(['name_idnet'],as_index=False).sum()
    df2 = df.drop_duplicates("name_idnet")[['speaker', 'name_idnet']]
    df2 = df2.merge(df1, on='name_idnet')
    df2 = df2.drop(["name_idnet"], axis=1)
    
    return df2


# In[ ]:


@app.route("/rasa_train_model",methods=["POST","GET"])
#This trains the RASA NLU model.

def train_rasa_model():
    nlu_training = "./rasa_byjus_test/data/nlu_byjus.md"
    training_data = load_data(nlu_training)
    trainer = Trainer(config.load("./rasa_byjus_test/config.yml"))
    interpreter = trainer.train(training_data)
    rasa_trained_model_directory = trainer.persist("./rasa_byjus_test/models", fixed_model_name="byjus_test_model")
    return rasa_trained_model_directory


# In[ ]:


@app.route("/rasa_model_output",methods=["POST","GET"])
# RASA predicted intents and converted to excel file.
#The output file directory is in current working directory.
def rasa_model_output(agent_customer_excel_file, confidence, rasa_trained_model_directory):
    interpreter = Interpreter.load(rasa_trained_model_directory)    
    predicted=[]
    result =  []
    no_intent_text = []
    no_intent_speaker = []
    i = 0
    for i in range(len(agent_customer_excel_file)):
        if confidence == None:
            confidence = 0.9
        a = interpreter.parse(agent_customer_excel_file.text[i])
        if a["intent_ranking"][0]["confidence"] > confidence:
            predicted.append(a["intent_ranking"][0]["name"])
        else:
            predicted.append("NO TAG")
            no_intent_text.append(agent_customer_excel_file.text[i])
            no_intent_speaker.append(agent_customer_excel_file.speaker[i])            
        no_intent = list(zip(no_intent_speaker, no_intent_text))
        no_intent = pd.DataFrame(no_intent, columns=["speaker", "text"])
        no_intent.to_excel("no_intent_classified.xlsx", index=None)
        result.append({"speaker": agent_customer_excel_file.speaker[i], "text":agent_customer_excel_file.text[i], "rasa_predicted":predicted[i]})

    agent_customer_excel_file["predicted"] = predicted
    rasa_output_file = agent_customer_excel_file.to_excel("rasa_predicted_output.xlsx", index=None)

    return jsonify(result)


# In[ ]:


@app.route("/byjus_introduction",methods=["POST","GET"])

def main_byjus():
    input_excel_file = "byjus_intro.xlsx"
    output_dir = os.getcwd() + "/agent_customer_seq_output.xlsx"

    df2 = agent_customer(input_excel_file, output_dir)
    confidence = request.args.get("confidence")
    rasa_trained_model_directory = "/home/chirag/Desktop/byjus_test_intro/rasa_byjus_test/models/byjus_test_model"
    rasa_output_file = rasa_model_output(df2, confidence, rasa_trained_model_directory)
    print("Check the excel file name : 'rasa_predicted_output.xlsx' \n in directory {}:".format(os.getcwd()))
    
    return rasa_output_file


# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


# In[ ]:




