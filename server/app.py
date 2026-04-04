import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from app.app import build_ui
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    app = build_ui()
    app.launch(server_name=server_name, server_port=7860, share=False)


if __name__ == "__main__":
    main()