from setuptools import find_namespace_packages, setup

# WHY find_namespace_packages and not the hand-written py_modules list this file
# carried before: that list named 18 modules out of the 36 the tree has, so
# ``pip install .`` produced an installation whose very first headless call died
# on ``ModuleNotFoundError: cards.cli`` -- the whole CLI surface, the MCP server,
# the budget and the notifier never shipped. Nobody noticed because the working
# venv is installed in develop mode and imports from the tree. A runner, which
# installs for real, notices immediately.
#
# ``processors``, ``flows`` and ``morph_mcp`` carry no __init__.py, so the plain
# find_packages would skip them; the namespace variant with an explicit include
# list takes them whole and adds nothing the include list does not name.
PACKAGES = find_namespace_packages(include=[
    "cards", "cards.*",
    "processors", "processors.*",
    "flows", "flows.*",
    "morph_mcp", "morph_mcp.*",
])

setup(
    name="mrph",
    version="1.0.56",
    packages=PACKAGES,
    py_modules=["mrph_console", "settings", "scheduler", "llm_dialog",
                "context_folder_dialog"],
    entry_points={"console_scripts": ["mrph = mrph_console:main"]},
    install_requires=['openai', 'anthropic', 'python-dotenv', 'ollama', 'setuptools',
                      'pysyun_conversation_flow@git+https://github.com/pysyun/pysyun_conversation_flow.git']
)
