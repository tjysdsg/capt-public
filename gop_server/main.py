import uvicorn
import os


def main():
    from gop_server.app import create_app
    app = create_app()
    server_port = int(os.environ.get('GOP_SERVER_PORT'))
    if server_port is None:
        server_port = 8080
    uvicorn.run(app, host="0.0.0.0", port=server_port)


if __name__ == "__main__":
    main()
