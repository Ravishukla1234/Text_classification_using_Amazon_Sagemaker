#app.py
import os
import requests
import json
from bs4 import BeautifulSoup
import re
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
def extract_text(webpage):
  html = requests.get(webpage).text
  text = BeautifulSoup(html, "lxml").text
  text = text.replace("\n"," ")
  text = re.sub('[^A-Za-z]+', ' ', text)
  text = re.sub('\W+',' ', text )
  return text.lower()

def predict_text(webpage,url):
    text = extract_text(webpage)
    request = {"inputs" : text}
    data = json.dumps(request)
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=data)
    #print( response.text)
    prediction = json.loads(json.loads(response.text)["body"])
    output = "Evergreen" if prediction["label"] == "LABEL_1" else "Ephemeral"
    return output


@app.route("/",methods = ["GET", "POST"])
def predict():
    url = "Amazon API Gateway url link"
    if(request.method == "POST"):
      webpage = request.form.get("webpage")
      print(f"webpage - {webpage}")
      if(webpage):
        output = predict_text(webpage,url )
        return 	render_template("index.html",prediction = output, webpage =webpage )	
    return render_template("index.html")

if __name__ =="__main__":
    app.run(port = 12000, debug = True)