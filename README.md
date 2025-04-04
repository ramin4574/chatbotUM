# University Student Assistant Chatbot

A Streamlit-based chatbot that helps students find information from university documents using AI.

## Features

- Document processing for PDF and TXT files
- AI-powered question answering
- Persistent vector store for efficient retrieval
- User-friendly interface
- Support for multiple document types (policies, courses, handbooks, FAQs)

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd university-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
   - Create a `.env` file in the root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

5. Create the required directories:
```bash
mkdir -p data/policies data/courses data/handbooks data/faqs
```

6. Add your documents to the appropriate directories:
   - Place policy documents in: `data/policies/`
   - Place course materials in: `data/courses/`
   - Place handbooks in: `data/handbooks/`
   - Place FAQs in: `data/faqs/`

## Running Locally

```bash
streamlit run app.py
```

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. Create a Streamlit Cloud account at https://streamlit.io/cloud
2. Connect your GitHub repository
3. Configure the deployment:
   - Set the main file to `app.py`
   - Add your OpenAI API key in the secrets section
4. Deploy!

### Option 2: Heroku

1. Create a `Procfile`:
```
web: streamlit run app.py --server.port $PORT
```

2. Create a `runtime.txt`:
```
python-3.9.x
```

3. Deploy to Heroku:
```bash
heroku create
git push heroku main
```

### Option 3: Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

2. Build and run:
```bash
docker build -t university-chatbot .
docker run -p 8501:8501 university-chatbot
```

## Security Considerations

- Never commit your `.env` file or API keys
- Use environment variables for sensitive information
- Consider implementing user authentication for production use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 