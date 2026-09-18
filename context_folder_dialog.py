import os
import sys
from datetime import datetime
from typing import List, Optional
from llm_dialog import LLMDialog


class ContextFolderDialog(LLMDialog):
    """Build a self-contained context dialog from files on disk.

    Two modes share one output shape (one ``user`` message per file, in the
    Morph 1.0 template):

    * **walk mode** (default) -- when ``file_list`` is ``None``, ``process``
      walks ``path`` and emits every file the ``filter_callback`` accepts. This
      is the original whole-project behaviour and is unchanged.
    * **file-list mode** -- when ``file_list`` is given (the ``context_slice``
      of a morph card, see ``documentation/batch-orchestrator.md``),
      ``process`` reads exactly those paths, relative to ``path``, in the order
      given, and the ``filter_callback`` is not consulted. A path that does not
      exist raises :class:`FileNotFoundError` naming it, so a mis-compiled slice
      fails loudly rather than silently dropping context.
    """

    def __init__(self, path, filter_callback=None, file_list: Optional[List[str]] = None):
        super().__init__()
        self.path = path
        self.filter_callback = filter_callback
        self.file_list = file_list

    def process(self, _):
        # Прогресс идёт в stderr, а не в stdout. WHY: эта же сборка контекста
        # лежит на пути безголовой команды (``mrph submit``/``run`` ->
        # ``cards.compiler``), чей контракт обещает, что stdout — ровно один
        # JSON-документ. Одна строка на файл среза ломала это обещание на
        # первом же реальном вызове, и ни один тест этого не видел: все
        # контрактные проверки падали ДО компиляции. Для человека в REPL ничего
        # не меняется — stderr идёт в тот же терминал.

        dialog = self

        if self.file_list is not None:
            for relative_path in self.file_list:
                file_path = os.path.join(self.path, relative_path)
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(
                        f"context_slice file not found: {file_path}"
                    )
                time = datetime.fromtimestamp(os.path.getmtime(file_path))
                with open(file_path, encoding='utf-8') as file:
                    print(f"Folder context file: {file_path}", file=sys.stderr)
                    file_contents = file.read()
                    self.assign("user", f"Contents for another file \"{file_path}\" in this project:\n\n---\n{file_contents}\n---\n", int(time.timestamp()) * 1000)
            return dialog

        for root, dirs, files in os.walk(self.path):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                if self.filter_callback and not self.filter_callback(file_path):
                    continue

                time = datetime.fromtimestamp(os.path.getmtime(file_path))
                with open(file_path, encoding='utf-8') as file:
                    print(f"Folder context file: {file_path}", file=sys.stderr)
                    file_contents = file.read()
                    self.assign("user", f"Contents for another file \"{file_path}\" in this project:\n\n---\n{file_contents}\n---\n", int(time.timestamp()) * 1000)

        return dialog
