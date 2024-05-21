from transformers import pipeline
import torch
import gradio as gr
import os

model_name = "deepset/roberta-base-squad2"
qna = pipeline("question-answering", model=model_name, tokenizer=model_name)
# img_qna = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

def load_file(file_input, encoding = 'utf-8'):  
  if not os.path.exists(file_input):
    raise FileNotFoundError(f"The file does not exist.")

  with open(file_input, 'r', encoding=encoding) as file:
    try:
      content = file.read()
    except UnicodeDecodeError:
      # If a UnicodeDecodeError occurs, try reading with 'latin1' encoding
      with open(file_input, 'r', encoding='latin1') as file:
        content = file.read()

  return content

def qna_content(content, question):
  answer = qna(question=question, context=content)
  # answer = img_qna(question=question, image=content)
  return answer

def answer_the_question_for_doc(text_input, file_input, question):
  #  Order of input parameters is Imp. for Gradio to accept respective Inputs 
  if file_input is not None:
    try:
      content = load_file(file_input)
    
    except FileNotFoundError as e:
      print(e)
      exit(1)
  else:
      content = text_input

  if not content or not question:
    return "Please provide both content and a question."

  answer = qna_content(content, question)
  return answer["answer"]

gr.close_all()

with gr.Blocks() as demo:
    gr.Markdown("# QnA System") 
    gr.Markdown("This App answers a question based on text content or uploaded file.")

    with gr.Row():
        text_input = gr.Textbox(label="Text Input", placeholder="Enter text content here...")
        file_input = gr.File(label="File Upload", file_types=['txt', 'png', 'jpeg', 'pdf'])

    question = gr.Textbox(label="Question", placeholder="Enter your question here...")
    output = gr.Textbox(label="Answer", placeholder="The answer will appear here...")

    text_input.change(lambda x: gr.update(visible=not x), inputs=text_input, outputs=file_input)
    file_input.change(lambda x: gr.update(visible=not x), inputs=file_input, outputs=text_input)

    button = gr.Button("Get Answer")
    button.click(answer_the_question_for_doc, inputs=[text_input, file_input, question], 
                outputs=output)
demo.launch()