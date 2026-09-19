"""The console entry point of the ``mrph`` command.

WHY this module exists at all, when ``bin/mrph`` already did the job: a
``scripts=`` entry is COPIED into the environment at install time, so a Morph
that morphs itself leaves the installed command at yesterday's copy until
somebody reinstalls by hand. A ``console_scripts`` entry point is a generated
launcher that imports this module at RUN time, so the installed command and the
tree never drift apart -- which is the whole point for an unattended run on a
machine nobody logs into.

The dispatch itself is unchanged and deliberately dumb: arguments mean the
headless surface, no arguments mean the console bot. ``bin/mrph`` is kept as a
one-line shim over this function, because "run it straight from a checkout" is
how the project is developed and that must not break.
"""

import sys
from typing import List, Optional

__all__ = ["main"]


def main(argv: Optional[List[str]] = None) -> int:
    """Run ``mrph`` and return the process exit code.

    ``argv`` is the argument list WITHOUT the program name; ``None`` reads
    ``sys.argv``. The imports live inside the branches on purpose: a scripted
    run must complete without the console bot -- and therefore without the
    conversation-flow dependency -- ever loading, which is the same discipline
    ``bin/mrph`` has followed since the headless surface existed.
    """
    from settings import load_settings

    load_settings()
    arguments = list(sys.argv[1:] if argv is None else argv)

    if arguments:
        from cards.cli import main as cli_main

        return cli_main(arguments)

    from flows.morph import MorphBot

    MorphBot(None).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
