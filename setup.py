from setuptools import setup

setup(
    name="mrph",
    version="1.0.44",
    scripts=["bin/mrph"],
    py_modules=["flows.morph", "settings", "llm_dialog", "context_folder_dialog", "ollama_processor"],
    install_requires=['openai', 'python-dotenv', 'ollama', 'setuptools',
                      'pysyun_conversation_flow@git+https://github.com/pysyun/pysyun_conversation_flow.git',
                      'python_claude_web_authenticator@git+https://github.com/pysyun/python_claude_web_authenticator'
                      '.git']
)
