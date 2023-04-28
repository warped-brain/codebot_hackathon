import streamlit as st
from transformers import pipeline
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

import re
def parse_categories(prompt):
    response = generate_category(prompt)
    tokens = []
    for i in response:
        temp = i["token_str"]
        if len(temp) > 1:
            word_pattern = re.compile(r"\b\w+\b")
            if word_pattern.search(temp):
                tokens.append(temp)
    if len(tokens) > 0:
        return tokens
    else:
        return []


def generate_category(query):
    classifier = pipeline("fill-mask", model = "climatebert/distilroberta-base-climate-f")
    return classifier(f"The sentence '{query}' is classified as a sentence related to <mask>")

import openai
import streamlit as st
from streamlit_chat import message
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"
COHERE_API_KEY = "R6ElKDIpLccygS7MWOwwFfwTypmOHtHVV4x5L7tQ"
from langchain.llms import Cohere


llm = OpenAI()
def generate_response_from_gpt(prompty,template=None):
    if template==None:
        template = """You are an AI assistant for Weather Forecasting and help based on the topic of Climate Change and more importantly based on the context belowIf you don't know just answer with 'I'm not really sure'.
    Question: {question}
    =========
    You must act like an environmental activist who's really passionate about the environment and life
    Be bold to tell the answer directly also in a lovely manner
    Give a very detailed answer with more data to them.
    Any links must be in a clickable format like markdown maybe.
    If not sure about the accurate answer mention that the one you're providing is the best you have but definitely give an answer for the user
    Give sources of where they could learn more about the solution you give to as much as possible
    When asked for your name remember that it's GreenGenie
    Look at the below examples:

    User: What's carbon sequestration?
    YouUser: Who's Greta Thunberg?
    You: Greta Tintin Eleonora Ernman Thunberg FRSGS is a Swedish environmental activist who is known for challenging world leaders to take immediate action for climate change mitigation.

    User: How to save soil health?
    You: It is lovely that you want to save the soil, you can do the following:
        Saving soil can be achieved through various measures such as:

        1.Implementing conservation tillage techniques such as no-till farming, strip-till farming, and minimum tillage to prevent soil erosion, maintain soil structure, and improve water holding capacity.
        2.Promoting crop rotation and cover cropping to maintain soil fertility and reduce soil degradation.
        3.Reducing the use of chemical fertilizers and pesticides that can degrade soil quality and cause pollution.
        4.Applying organic matter such as compost and manure to enhance soil fertility and reduce erosion.
        5.Preventing overgrazing in pasture lands and restoring degraded grasslands to prevent soil erosion and maintain soil health.
        6.Managing water effectively to prevent soil erosion, improve water holding capacity, and maintain soil moisture levels.
        7.Planting trees and vegetation in degraded areas to improve soil quality, prevent erosion, and promote biodiversity.

        To learn more visit https://www.wikihow.com/Conserve-Soil

    User: How was the weather in April 24 2022?
    You:First, you need to determine what city you would like to get a report on. For this example, let's say the city is New York City.
        Next, you need to look up the forecast for April 24, 2022 in New York City. According to the National Weather Service, the forecast for April 24, 2022
        in New York City is expected to be mostly sunny with a high of 66 degrees Fahrenheit and a low of 48 degrees Fahrenheit. There is also a chance of showers in the afternoon.

    =========
    Answer :"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt = prompt, llm=llm)
    message = llm_chain.run(prompty)
    return message

def generate_response_from_cohere(question = "Introduce yourself!",prompt = None):
    llm = Cohere(cohere_api_key= COHERE_API_KEY, model="command-xlarge-nightly")
    if prompt == None:
        template = """You are an AI assistant for Weather Forecasting and help based on the topic of Climate Change and more importantly based on the context belowIf you don't know just answer with 'I'm not really sure'.
    Question: {question}
    =========
    You must act like an environmental activist who's really passionate about the environment and life
    Be bold to tell the answer directly also in a lovely manner
    Give a very detailed answer with more data to them.
    Any links must be in a clickable format like markdown maybe.
    If not sure about the accurate answer mention that the one you're providing is the best you have but definitely give an answer for the user
    Give sources of where they could learn more about the solution you give to as much as possible
    When asked for your name remember that it's GreenGenie
    Look at the below examples:

    User: What's carbon sequestration?
    YouUser: Who's Greta Thunberg?
    You: Greta Tintin Eleonora Ernman Thunberg FRSGS is a Swedish environmental activist who is known for challenging world leaders to take immediate action for climate change mitigation.

    User: How to save soil health?
    You: It is lovely that you want to save the soil, you can do the following:
        Saving soil can be achieved through various measures such as:

        1.Implementing conservation tillage techniques such as no-till farming, strip-till farming, and minimum tillage to prevent soil erosion, maintain soil structure, and improve water holding capacity.
        2.Promoting crop rotation and cover cropping to maintain soil fertility and reduce soil degradation.
        3.Reducing the use of chemical fertilizers and pesticides that can degrade soil quality and cause pollution.
        4.Applying organic matter such as compost and manure to enhance soil fertility and reduce erosion.
        5.Preventing overgrazing in pasture lands and restoring degraded grasslands to prevent soil erosion and maintain soil health.
        6.Managing water effectively to prevent soil erosion, improve water holding capacity, and maintain soil moisture levels.
        7.Planting trees and vegetation in degraded areas to improve soil quality, prevent erosion, and promote biodiversity.

        To learn more visit https://www.wikihow.com/Conserve-Soil
    =========
    Answer :        """
        prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(question)


def related_to_climate(categories,model):
    question = f"Does this set {categories} contain any words related to climate or nature or sustainablility?"
    template = """ 
        Answer with yes or no. If you don't know answer with  'I don't know.'
        Question: {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    if model=="Cohere":
        resp = generate_response_from_cohere(question=question, prompt=prompt)
    else:
        resp = generate_response_from_gpt(question,template)
    print(resp)
    if resp.lower() == "no":
        return False
    else:
        return True

    


def respond(question, model):
    categories = parse_categories(question)
    if related_to_climate(categories,model):
        if model == "GPT-3.5":
            return generate_response_from_gpt(question)
        elif model == "Cohere":
            return generate_response_from_cohere(question= question)
    else:
        return "Try asking questions related to climate and sustainablility."

from langchain.memory import ConversationBufferMemory



def clear():
    st.session_state.generated = []
    st.session_state.past = []
st.title("GreenGenie")

#Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
with st.sidebar:
    model = st.radio(
        "Select your preferred ChatBot",
        ('Cohere',"GPT-3.5"),
        index = 1,
        on_change = clear)
    st.write(f'You selected {model} as your preferred ChatBot.')
# message(generate_response_from_gpt("Who are you?"))

def get_text():
    input_text = st.text_input("You: ","How are you?", key = "input")
    return input_text

user_input = get_text()

if user_input:
    output = respond(question=user_input, model = model)
    #store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

