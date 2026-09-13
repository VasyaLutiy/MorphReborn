from setuptools import setup

setup(
    name="mrph",
    version="1.0.55",
    scripts=["bin/mrph"],
    py_modules=["flows.morph", "settings", "scheduler", "llm_dialog", "context_folder_dialog",
                "processors.registry", "processors.ollama_processor",
                "processors.llama_cpp_processor", "processors.openai_processor",
                "processors.anthropic_processor", "processors.batch"],
    install_requires=['openai', 'anthropic', 'python-dotenv', 'ollama', 'setuptools',
                      'pysyun_conversation_flow@git+https://github.com/pysyun/pysyun_conversation_flow.git']
)
