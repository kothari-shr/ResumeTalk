# Resume Aware Chatbot - Streamlit Deployment

This guide shows how to deploy your Resume Aware Chatbot on Streamlit Cloud or Hugging Face Spaces.

## 🚀 Quick Start (Local)

```bash
# Install streamlit dependencies
pip install -r requirements-streamlit.txt

# Run the app
streamlit run streamlit_app.py
```

## ☁️ Deploy to Streamlit Cloud (FREE)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit interface"
   git push
   ```

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `kothari-shr/langchain-llm-learning`
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Add Secrets** (Environment Variables)
   - In Streamlit Cloud dashboard, go to your app settings
   - Click "Secrets" 
   - Add your secrets in TOML format:
   ```toml
   OPENAI_API_KEY = "sk-..."
   RESUME_PATH = "resume.pdf"
   LLM_MODEL = "gpt-4o-mini"
   ```

## 🤗 Deploy to Hugging Face Spaces (FREE)

1. **Create a new Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Name it (e.g., "resume-chatbot")

2. **Upload your files**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/resume-chatbot
   cd resume-chatbot
   
   # Copy your files from your repo
   cp -r /path/to/langchain-llm-learning/* .
   
   # Create packages.txt if needed (for system dependencies)
   echo "poppler-utils" > packages.txt
   
   git add .
   git commit -m "Initial commit"
   git push
   ```

3. **Add Secrets**
   - In Space settings, go to "Settings" > "Repository secrets"
   - Add:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `RESUME_PATH`: Path to your resume in the repo

4. **Note**: The existing `requirements.txt` already contains all necessary dependencies

## 📋 Pre-Deployment Checklist

- [ ] `.env` file is in `.gitignore` (never commit secrets!)
- [ ] Resume PDF is added to the repository
- [ ] Update `RESUME_PATH` in secrets to match your resume location
- [ ] OpenAI API key is added to secrets
- [ ] Test locally with `streamlit run streamlit_app.py`

## 🔧 Configuration

All settings are managed through environment variables or Streamlit secrets:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `RESUME_PATH`: Path to your resume PDF
- `LLM_MODEL`: Model to use (default: gpt-4o-mini)
- `LLM_TEMPERATURE`: Temperature setting (default: 0.2)
- `RETRIEVER_K`: Number of documents to retrieve (default: 4)

## 📁 File Structure for Deployment

```
ResumeTalk/
├── streamlit_app.py          # Main Streamlit app
├── requirements.txt          # Dependencies (rename from requirements-streamlit.txt for deployment)
├── .streamlit/
│   └── config.toml           # Streamlit config
├── app/                      # Your existing app code
│   ├── api/
│   ├── core/
│   ├── models/
│   └── services/
├── resume_loader.py
├── rag_chain.py
├── email_sender.py
├── main.py
└── resume.pdf                # Your resume
```

## 🎨 Features

- 💬 Chat interface with conversation history
- 📄 Source document viewer
- 🗑️ Clear chat history
- 📱 Responsive design
- 🎨 Custom theming

## 🐛 Troubleshooting

### App won't start
- Check that all required environment variables are set
- Verify `RESUME_PATH` points to an existing file
- Check logs for specific error messages

### "RAG service not initialized"
- Ensure your resume PDF exists at the specified path
- Check that OpenAI API key is valid
- Look for initialization errors in logs

### Import errors
- Make sure all dependencies are in `requirements.txt`
- For Streamlit Cloud, dependencies install automatically
- Verify streamlit is included in `requirements.txt`

## 💡 Tips

1. **Free Tier Limits**: 
   - Streamlit Cloud: Free tier has usage limits
   - Hugging Face: Free tier works great for personal projects

2. **Performance**: 
   - First load takes time to initialize FAISS index
   - Consider caching the vectorstore if deploying frequently

3. **Cost**: 
   - Hosting is free on both platforms
   - You only pay for OpenAI API usage

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
