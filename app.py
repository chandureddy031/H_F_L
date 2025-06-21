from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ChatBot:
    def __init__(self, model_path="./fine_tuned_model"):
        logger.info("Initializing chatbot...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device.upper()}")
        
        try:
            # Check if fine-tuned model exists
            adapter_config = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(model_path) and os.path.exists(adapter_config):
                logger.info("🎉 Found fine-tuned model! Loading...")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/DialoGPT-small",
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                
                # Load fine-tuned adapter
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.is_finetuned = True
                logger.info("✅ Fine-tuned model loaded successfully!")
            else:
                logger.warning("⚠️ Fine-tuned model not found. Using base model.")
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/DialoGPT-small",
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                self.is_finetuned = False
                logger.info("✅ Base model loaded successfully!")
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
            self.is_finetuned = False
            self.model = None
            self.tokenizer = None

    def generate_response(self, user_input, max_new_tokens=100):
        if not self.model or not self.tokenizer:
            return "Model not loaded. Please check server logs."
        
        try:
            # Clean input
            user_input = user_input.strip()
            if not user_input:
                return "Please type a message..."
            
            # Format prompt
            prompt = f"{user_input}{self.tokenizer.eos_token}"
            
            # Tokenize
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            
            if self.device == "cuda":
                inputs = inputs.cuda()
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            response = full_response[len(prompt):].strip()
            
            return response or "I understand. Could you tell me more?"
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return "I'm having trouble thinking right now. Please try again."

# Initialize chatbot
chatbot = ChatBot()

# HTML template with guaranteed status display
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Custom AI Assistant</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
            color: #333;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .chat-container {
            width: 100%;
            max-width: 800px;
            background: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
            display: flex;
            flex-direction: column;
            height: 90vh;
        }
        
        .chat-header {
            background: linear-gradient(to right, #4b6cb7, #182848);
            color: white;
            padding: 25px 20px;
            text-align: center;
            position: relative;
        }
        
        .chat-header h1 {
            font-size: 1.8rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .status-container {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 8px 15px;
            display: inline-block;
            margin-top: 10px;
            font-weight: 500;
            backdrop-filter: blur(5px);
        }
        
        .status-connected {
            color: #aaffaa;
        }
        
        .status-base {
            color: #ffcc66;
        }
        
        .status-error {
            color: #ff6666;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .message {
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 18px;
            line-height: 1.5;
            position: relative;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background: #4b6cb7;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 5px;
        }
        
        .bot-message {
            background: #e9ecef;
            align-self: flex-start;
            border-bottom-left-radius: 5px;
        }
        
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .input-group {
            display: flex;
            gap: 12px;
        }
        
        #userInput {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 30px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #userInput:focus {
            border-color: #4b6cb7;
        }
        
        button {
            padding: 15px 30px;
            background: #4b6cb7;
            color: white;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        button:hover {
            background: #3a5795;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .welcome-message {
            text-align: center;
            color: #6c757d;
            font-style: italic;
            margin-top: 10px;
        }
        
        @media (max-width: 768px) {
            .chat-container {
                height: 95vh;
                border-radius: 15px;
            }
            
            .chat-header h1 {
                font-size: 1.5rem;
            }
            
            .message {
                max-width: 90%;
            }
            
            button {
                padding: 15px 20px;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 Custom AI Assistant</h1>
            <div class="status-container" id="model-status">
                Loading model status...
            </div>
        </div>
        
        <div class="chat-messages" id="messages">
            <div class="message bot-message" id="welcome-message">
                Initializing AI assistant...
            </div>
        </div>
        
        <div class="input-container">
            <div class="input-group">
                <input type="text" id="userInput" placeholder="Type your message here..." autocomplete="off">
                <button onclick="sendMessage()">Send</button>
            </div>
            <div class="welcome-message">
                Powered by fine-tuned DialoGPT model
            </div>
        </div>
    </div>

    <script>
        // DOM elements
        const userInput = document.getElementById('userInput');
        const messagesDiv = document.getElementById('messages');
        const statusElement = document.getElementById('model-status');
        const welcomeMessage = document.getElementById('welcome-message');
        
        // Update status display
        function updateStatus(status, type = '') {
            statusElement.textContent = status;
            statusElement.className = 'status-container';
            if (type) {
                statusElement.classList.add(`status-${type}`);
            }
        }
        
        // Add message to chat
        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return messageDiv;
        }
        
        // Send message to server
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            userInput.value = '';
            userInput.disabled = true;
            
            // Add temporary bot message
            const botMessage = addMessage('Thinking...', 'bot');
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                
                const data = await response.json();
                
                // Update bot message
                botMessage.textContent = data.response;
                
            } catch (error) {
                botMessage.textContent = "I'm having trouble connecting. Please try again.";
            } finally {
                userInput.disabled = false;
                userInput.focus();
            }
        }
        
        // Check model status on load
        async function checkModelStatus() {
            updateStatus('Checking model status...');
            
            try {
                const response = await fetch('/status');
                const status = await response.json();
                
                if (status.model_type === 'fine-tuned') {
                    updateStatus('🎉 Using fine-tuned model', 'connected');
                    welcomeMessage.textContent = "Hello! I'm your custom-trained AI assistant. How can I help you today?";
                } else {
                    updateStatus('⚠️ Using base model', 'base');
                    welcomeMessage.textContent = "Hello! I'm the base AI model. For better responses, train me with your data.";
                }
            } catch (error) {
                updateStatus('❌ Status check failed', 'error');
                welcomeMessage.textContent = "Hello! I'm your AI assistant. How can I help you today?";
            }
        }
        
        // Initialize chat
        function initChat() {
            // Set up event listeners
            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });
            
            document.querySelector('button').addEventListener('click', sendMessage);
            
            // Focus input field
            userInput.focus();
            
            // Check model status
            checkModelStatus();
        }
        
        // Start when page loads
        window.addEventListener('DOMContentLoaded', initChat);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    return jsonify({
        'model_type': 'fine-tuned' if chatbot.is_finetuned else 'base',
        'ready': True
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate response
        response = chatbot.generate_response(user_message)
        
        return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'response': "I'm having trouble thinking right now. Please try again."})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    logger.info("Starting chat application...")
    logger.info("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000)