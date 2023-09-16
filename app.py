from flask import Flask, render_template, request, jsonify
from ctransformers import AutoModelForCausalLM

app = Flask(__name__)

# Load the LLM model and tokenizer during application startup
llm = AutoModelForCausalLM.from_pretrained(
    "codellama-7b-instruct.Q3_K_M.gguf",
    model_type='llama',
    max_new_tokens=510,
    repetition_penalty=1.13,
    temperature=0.1
)

# Function to generate responses
def llm_function(message, model):
    response = model(
        message
    )
    output_text = response[0]['generated_text']
    return output_text

@app.route('/')
def chat_page():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_response():
    prompt = request.form['prompt']
    # Generate a response using the preloaded LLM model
    output_text = llm_function(prompt, llm)
    return jsonify({'response': output_text})

if __name__ == '__main__':
    app.run(debug=True)
