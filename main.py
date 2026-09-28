"""Entry point: `python main.py` opens the app; `python main.py <file>` converts from the command line."""

import sys


def main():
    if len(sys.argv) > 1:
        from engine.cli import main as cli_main
        sys.exit(cli_main())
    from gui.app import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
