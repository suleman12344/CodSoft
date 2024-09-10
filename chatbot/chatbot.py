import re
import nltk
from nltk.chat.util import Chat, reflections
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#pairs data set get from the kaggle
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how are you today?"]
    ],
    [
        r"(.*)(help|assist)(.*)",
        ["I can help you. What do you need assistance with?", "Sure, I'm here to help!"]
    ],
    [
        r"what is your name?",
        ["ou can just call me robot. I'm a chatbot."]
    ],
    [
        r"how are you(.*)?",
        ["I'm doing very well, thank you!", "I'm great! How about you?"]
    ],
    [
        r"sorry(.*)",
        ["It's alright", "No worries, it's okay."]
    ],
    [
        r"I'm(.*) (good|well|okay|ok)",
        ["Nice to hear that!", "Glad to know you're doing well!"]
    ],
    [
        r"(hi|hey|hello|hola|holla)(.*)",
        ["Hello!", "Hey there!", "Hi, how can I assist you today?"]
    ],
    [
        r"what (.*) want?",
        ["I want to help you in any way I can.", "I'm here to chat and assist you."]
    ],
    [
        r"who created you?",
        ["why you asking that?","I am created by Muhammad Suleman"]
    ],
    [
        r"where are you from?",
        ["I am from technology world, but created in pakistan"]
    ],
    [
        r"is it raining in (.*)",
        ["I'm not sure about the weather in %1, but I can check for you!", "In %1, there's a 50% chance of rain."]
    ],
    [
        r"how (.*) health (.*)",
        ["Health is very important! But as a computer, I don't need to worry about that."]
    ],
    [
        r"(.*)(sports|game|sport)(.*)",
        ["I'm a big fan of cricket!", "I love discussing sports, especially cricket."]
    ],
    [
        r"what time is it?",
        ["It's chatbot time!", "I'm not equipped with a clock, but it's always a good time to chat."]
    ],
    [
        r"(.*)(weather|temperature)(.*)",
        ["The weather is great today!", "It's a bit chilly right now."]
    ],
    [
        r"(.*)",
        ["Can you please rephrase that?", "I'm not sure I understand. Could you elaborate?"]
    ]
]

class chatbot:
  def __init__(self,pairs):
    self.chat = Chat(pairs,reflections)
  def respond(self,user_input):
    return self.chat.respond(user_input)


def chat_with_bot():
  bot = chatbot(pairs)
  print("Ask what you wanna ask or Type 'Quite' to quit")
  while True:
    user_input = input('User: ' )
    if user_input.lower() == 'quite':
      print("thank you, See you again")
      break
    else:
      print("Bot: ", bot.respond(user_input))
chat_with_bot()
