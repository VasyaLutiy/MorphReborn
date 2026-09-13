from setuptools import setup

setup(
    name="mrph",
    version="1.0.54",
    scripts=["bin/mrph"],
    py_modules=["flows.morph", "settings", "scheduler", "llm_dialog", "context_folder_dialog",
                "processors.registry", "processors.ollama_processor",
                "processors.llama_cpp_processor", "processors.openai_processor"],
    install_requires=['openai', 'python-dotenv', 'ollama', 'setuptools',
                      'pysyun_conversation_flow@git+https://github.com/pysyun/pysyun_conversation_flow.git']
)
